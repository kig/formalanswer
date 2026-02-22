import subprocess
import json
import os
import time
from typing import Optional, Dict, Any
from .common import VerificationResult

class LeanServer:
    """
    Manages a persistent Lean 4 server process to minimize verification latency.
    Uses Lean's JSON-RPC interface.
    """
    def __init__(self, work_dir: str = "work"):
        self.work_dir = work_dir
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    def start(self):
        """Starts the Lean server in the background."""
        if self.process and self.process.poll() is None:
            return # Already running
            
        cmd = ["./env_wrapper.sh", "lake", "env", "lean", "--server"]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Give it a moment to initialize
        time.sleep(1)
        if self.process.poll() is not None:
            stderr_out = self.process.stderr.read()
            print(f"Lean server failed to start: {stderr_out}")

    def stop(self):
        """Stops the Lean server."""
        if self.process:
            self.process.terminate()
            self.process = None

    def _send_request(self, method: str, params: Dict[str, Any]):
        if not self.process or self.process.poll() is not None:
            self.start()
            
        self.request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        raw_req = json.dumps(req)
        header = f"Content-Length: {len(raw_req)}\r\n\r\n"
        try:
            self.process.stdin.write(header + raw_req)
            self.process.stdin.flush()
        except BrokenPipeError:
            print("Broken pipe when sending to Lean server. Attempting restart.")
            self.start()
            self.process.stdin.write(header + raw_req)
            self.process.stdin.flush()

    def _read_response(self) -> Dict[str, Any]:
        line = self.process.stdout.readline()
        if not line or not line.startswith("Content-Length:"):
            return {}
        
        try:
            content_length = int(line.split(":")[1].strip())
            self.process.stdout.readline() # Read the \r\n
            content = self.process.stdout.read(content_length)
            return json.loads(content)
        except Exception as e:
            print(f"Error reading Lean server response: {e}")
            return {}

    def _send_notification(self, method: str, params: Dict[str, Any]):
        if not self.process or self.process.poll() is not None:
            self.start()
            
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        raw_req = json.dumps(req)
        header = f"Content-Length: {len(raw_req)}\r\n\r\n"
        try:
            self.process.stdin.write(header + raw_req)
            self.process.stdin.flush()
        except BrokenPipeError:
            self.start()
            self.process.stdin.write(header + raw_req)
            self.process.stdin.flush()

    def verify_snippet(self, code: str) -> VerificationResult:
        """
        Verifies a snippet by simulating an 'open file' and checking diagnostics.
        """
        if not self.process or self.process.poll() is not None:
            self.start()

        file_uri = "file:///tmp/temp.lean" # Real-looking URI
        
        # 1. didOpen
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_uri,
                "languageId": "lean4",
                "version": int(time.time()),
                "text": code
            }
        })
        
        # 2. didSave (triggers full elaboration in many LSP servers)
        self._send_notification("textDocument/didSave", {
            "textDocument": {"uri": file_uri}
        })
        
        # 3. Wait for diagnostics
        start_time = time.time()
        errors = []
        # Polling loop: Wait for a 'textDocument/publishDiagnostics' response
        while time.time() - start_time < 5: 
            resp = self._read_response()
            if not resp:
                time.sleep(0.1)
                continue
                
            if resp.get("method") == "textDocument/publishDiagnostics":
                if resp["params"]["uri"] == file_uri:
                    diags = resp["params"]["diagnostics"]
                    # If we got an empty list of diagnostics, it might mean success
                    # OR it might mean it hasn't finished yet.
                    # Usually, errors are reported quickly.
                    for d in diags:
                        if d.get("severity", 1) <= 1: # 1=Error, 2=Warning
                            errors.append(f"Line {d['range']['start']['line']}, Col {d['range']['start']['character']}: {d['message']}")
                    
                    if not diags: # Explicitly empty diags for our file = likely clean
                         break
                    else:
                         # We have diags (errors or warnings).
                         break
            
        # 4. didClose
        self._send_notification("textDocument/didClose", {
            "textDocument": {"uri": file_uri}
        })

        if errors:
            return VerificationResult(False, "Lean verification failed", "\n".join(errors))
        
        if "sorry" in code:
            return VerificationResult(False, "Lean code contains 'sorry' (incomplete proof)")
            
        return VerificationResult(True, "Lean verification successful")

# Global instance for easy reuse
_global_lean_server = None

def get_lean_server():
    global _global_lean_server
    if _global_lean_server is None:
        _global_lean_server = LeanServer()
        _global_lean_server.start()
    return _global_lean_server