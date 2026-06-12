import http.server
import json
import socketserver
import os
import sys

# Add core_ai to path so we can import binding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core_ai')))

PORT = 4040
DIRECTORY = "ui"

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

            # For testing purposes, if engine is not available, return mock response
            # In production, you'd initialize AIEngine here or globally

            try:
                # Attempt to use the actual engine if configured via environment variables
                lib_path = os.environ.get('AI_LIB_PATH')
                model_path = os.environ.get('AI_MODEL_PATH')

                if lib_path and model_path:
                    from binding import AIEngine
                    engine = AIEngine(lib_path, model_path)
                    for token in engine.generate_stream(prompt):
                        self.wfile.write(f"{len(token):x}\r\n{token}\r\n".encode())
                        self.wfile.flush()
                else:
                    mock_resp = f"<think>Analyzing request: {prompt}</think>This is a mock response because AI_LIB_PATH and AI_MODEL_PATH are not set. Compile the library and set the env vars to use the real engine."
                    for word in mock_resp.split(' '):
                        chunk = word + ' '
                        self.wfile.write(f"{len(chunk):x}\r\n{chunk}\r\n".encode())
                        self.wfile.flush()
            except Exception as e:
                err_msg = f"Error: {str(e)}"
                self.wfile.write(f"{len(err_msg):x}\r\n{err_msg}\r\n".encode())

            self.wfile.write(b"0\r\n\r\n")
        else:
            super().do_POST()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"Serving UI at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
