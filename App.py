from flask import Flask,render_template
from flask_mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_HOST'] = 'n0m3l0'
app.config['MYSQL_DB'] = 'ortopeñadb'
mysql = MySQL(app)

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/add_contact')
def add_contact():
    return 'Agregar Contacto'

@app.route('/edit')
def edit_contact():
    return 'Editar contacto'


@app.route('/delete')
def delete_contact():
    return 'Eliminar Contacto'

if __name__ == '__main__':
    app.run(port = 3000, debug=True)
