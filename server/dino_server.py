import http.server as https
import socketserver as ss

class QuietHandler(https.SimpleHTTPRequestHandler):
    """Serve files without logging each request to stdout."""
    def log_message(self, format, *args):
        pass

class HTTPServer:
    def __init__(self, port, game_dir):
        self.port = port
        self.game_dir = game_dir

    def start_http_server(self):
        handler = QuietHandler
        # Change working directory so HTTPServer picks up GAME_DIR
        ss.TCPServer.allow_reuse_address = True
        with ss.TCPServer(("", self.port), handler) as httpd:
            httpd.RequestHandlerClass.directory = str(self.game_dir)
            print(f"📡 HTTP server serving {self.game_dir} at http://localhost:{self.port}/")
            httpd.serve_forever()


