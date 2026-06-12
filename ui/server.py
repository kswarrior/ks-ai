import http.server
import json
import socketserver
import os
import sys

# Add core_ai to path so we can import binding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core_ai')))

PORT = 4040
DIRECTORY = "ui"

# Global engine instance to avoid reloading model on every request
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from binding import AIEngine
        # AIEngine now handles its own auto-detection of lib and model
        _engine = AIEngine()
    return _engine

class TestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            prompt = data.get('prompt', '')

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()

            try:
                engine = get_engine()
                for token in engine.generate_stream(prompt):
                    # HTTP Chunked encoding: [length in hex]\r\n[data]\r\n
                    chunk = token.encode('utf-8')
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except Exception as e:
                err_msg = f"\n[Backend Error] {str(e)}".encode('utf-8')
                self.wfile.write(f"{len(err_msg):x}\r\n".encode())
                self.wfile.write(err_msg)
                self.wfile.write(b"\r\n")

            # End of chunks
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            super().do_POST()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"\033[92m[Server]\033[0m UI running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            sys.exit(0)
