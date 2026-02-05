class VerificationResult:
    def __init__(self, success, message, details=""):
        self.success = success
        self.message = message
        self.details = details

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        return f"[{status}] {self.message}\n{self.details}"
