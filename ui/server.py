import http.server
import json
import socketserver
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core_ai')))

PORT = 4040
DIRECTORY = "ui"

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        try:
            from binding import AIEngine
            print("\033[92m[Server]\033[0m Initializing AIEngine...")
            _engine = AIEngine()
        except Exception as e:
            print(f"\033[91m[Server Error]\033[0m {e}")
    return _engine

class TestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/generate':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                prompt = data.get('prompt', '')

                print(f"\033[92m[Server]\033[0m New request: {prompt[:30]}...")

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()

                engine = get_engine()
                if engine:
                    count = 0
                    for token in engine.generate_stream(prompt):
                        chunk = token.encode('utf-8')
                        self.wfile.write(f"{len(chunk):x}\r\n".encode())
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        count += 1
                    print(f"\033[92m[Server]\033[0m Stream finished. {count} tokens sent.")
                else:
                    err = "Error: AI engine not initialized.".encode('utf-8')
                    self.wfile.write(f"{len(err):x}\r\n{err}\r\n".encode())

                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception as e:
                print(f"\033[91m[Server Error]\033[0m {e}")
        else:
            super().do_POST()

if __name__ == "__main__":
    # Ensure engine is loaded before serving
    get_engine()

    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"\033[92m[Server]\033[0m Listening at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            sys.exit(0)
