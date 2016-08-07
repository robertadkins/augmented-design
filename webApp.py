import random
import string

from bottle import route, run, static_file

@route('/hello')
def hello():
    return "Hello World!"

@route('/')
def hello():
    filename='index.html'
    return static_file(filename, root='/')

run(host='localhost', port=8080, debug=True)
def generateFilename () :
  return "".join( [random.choice(string.letters) for i in xrange(6)] )

#@route('/')
def server_index():
    return "HI";
#    filename='index.html'
#    return static_file(filename, root='/')

@route('/design', method='POST')
def upload_design():
    upload = request.files.get('upload')
    filename = "res/design"+generateFilename()+".jpg"
    upload.save(filename)
    return filename

@route('/image', method='POST')
def upload_design():
    upload = request.files.get('upload')
    filename = "res/image"+generateFilename()+".jpg"
    upload.save(filename)
    return filename

@route('/preview', method='POST')
def request_preview():
    pass

if __name__ == '__main__':
    run(host='localhost', port=8080)
