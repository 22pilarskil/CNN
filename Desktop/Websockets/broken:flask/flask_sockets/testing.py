from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)

app.config["SECRET_KEY"] = "KittyCat123"
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('/testing.html')


@socketio.on('connected')
def yeet(data):
    print('recieved something: '+str(data))
    socketio.emit('response', data)

if __name__ == '__main__':
    socketio.run(app, debug=True)

