"""
SWYNEX Internship - Task 3: simple web interface

Paste a customer email into the box and the page shows the predicted category,
the confidence, and whether the email needs human review.

How to run:
    python app.py
Then open http://localhost:8000 in your browser.

Only the Python standard library is used for the web page, so there is nothing
extra to install besides scikit-learn.
"""

import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from email_classifier import REVIEW_THRESHOLD, build_model, classify

MODEL = build_model()

PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support Email Classifier</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 640px; margin: 30px auto; padding: 0 16px; }}
    textarea {{ width: 100%; height: 130px; font-size: 16px; padding: 8px; box-sizing: border-box; }}
    button {{ margin-top: 10px; padding: 10px 20px; font-size: 16px; }}
    .result {{ margin-top: 20px; padding: 14px; border-radius: 8px; background: #eef3ff; }}
    .review {{ background: #fff3d6; }}
  </style>
</head>
<body>
  <h2>Customer Support Email Classifier</h2>
  <p>Paste an email below. Categories: Billing, Technical Issue,
     Refund / Cancellation, General Inquiry.</p>
  <form method="post">
    <textarea name="email" placeholder="Type or paste a customer email here">{email}</textarea>
    <br><button type="submit">Classify</button>
  </form>
  {result}
</body>
</html>
"""


def render(email="", result_html=""):
    return PAGE.format(email=html.escape(email), result=result_html)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(render())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        email = form.get("email", [""])[0].strip()

        if not email:
            self._send(render(result_html="<p>Please enter an email first.</p>"))
            return

        category, confidence, needs_review = classify(MODEL, email)
        if needs_review:
            result_html = (
                '<div class="result review">'
                "<b>Needs human review</b><br>"
                f"The model is not confident (best guess: {html.escape(category)}, "
                f"{confidence:.0%}; threshold {REVIEW_THRESHOLD:.0%}).</div>"
            )
        else:
            result_html = (
                '<div class="result">'
                f"<b>Category:</b> {html.escape(category)}<br>"
                f"<b>Confidence:</b> {confidence:.0%}</div>"
            )
        self._send(render(email, result_html))


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Open http://localhost:8000 in your browser. Press Ctrl+C to stop.")
    server.serve_forever()
