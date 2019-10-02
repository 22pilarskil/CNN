import mysql.connector
import datetime

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="KittyCat123",
    database="KIOSK"
)

######DATA STORED AS TUPLESSSSSS
mycursor = mydb.cursor()

#formula = "INSERT INTO Students (id, name, privilege, pathToFile, grade) VALUES (%s, %s, %s, %s, %s)"
#mycursor.execute (formula, [12345, "Liam", "None", "yeet", 10])
#mydb.commit()

mycursor.execute("SELECT * FROM Students WHERE id = 16103")

def getTableNames():
    tbs = []
    mycursor.execute("SHOW TABLES")
    for tb in mycursor:
        tbs.append(tb[0])
    return tbs


def addData():
    sqlFormula = ("INSERT INTO Students (id, name, privilege, pathToFile, grade) VALUES (:id, :name, :privilege, :pathToFile, :grade)", {"id": 12345, "name": "Liam Pilarski", "privilege": False, "pathToFile":"yeet", "grade": 10})
    data = ("Lukas", 99)
    mycursor.execute(sqlFormula)
    mydb.commit()

addData()

def getData():
    mycursor.execute("SELECT * FROM Students WHERE student LIKE 'l%'")
    print(mycursor.fetchone())

def updateData():
    mycursor.execute("UPDATE Students SET score = 98 WHERE student = 'Liam'")
    mydb.commit()

def deleteData():
    mycursor.execute("DELETE FROM Students WHERE NOT(student LIKE 'l%')")
    mydb.commit()

def deleteTable():
    mycursor.execute("DROP TABLE IF EXISTS Students")
    mydb.commit()

#addTable("Studentswithage")
#addData()
#updateData()
#getData()
#deleteData()
#deleteTable()
