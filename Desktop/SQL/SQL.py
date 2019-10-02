import mysql.connector
from datetime import *

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="KittyCat123",
    database="KIOSK",
)
hi = {}
######DATA STORED AS TUPLESSSSSS
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM Users WHERE uid = 'hi'")
print(len(mycursor.fetchall()))
print(mycursor.fetchall())
ID=False
formula = "SELECT * from students where "+("id" if ID else "grade")+" <11"
mycursor.execute(formula)
print(mycursor.fetchall())

hour = "11"
minute = "13"
second = "6"
startTime = datetime.strptime(hour+":"+minute+":"+second, '%H:%M:%S')
endTime = datetime.strptime('11:13:15', '%H:%M:%S')
formula = "SELECT TIMESTAMPDIFF(day,\'%s\',\'%s\')" %(startTime, endTime)
mycursor.execute(formula)
dif = mycursor.fetchall()[0][0]
print(dif)
base = "SELECT id FROM Transactions WHERE time_ < \'%s\' AND time_ > \'%s\'" %(endTime, startTime)
print(base)
mycursor.execute(base)
print(mycursor.fetchall())
'''
file = open("/Users/michaelpilarski/Desktop/kioskDB.sql", 'r')
for line in file:
    mycursor.execute(line)
print("done")
'''
#addTable()

#addTable("Students")
def getTableNames():
    tbs = []
    mycursor.execute("SHOW TABLES")
    for tb in mycursor:
        tbs.append(tb[0])
    return tbs

def addData():
    sqlFormula = "INSERT INTO Students (student, score) VALUES (%s, %s)"
    data = ("Lukas", 99)
    mycursor.execute(sqlFormula, data)
    mydb.commit()

def getData():
    mycursor.execute("SELECT * FROM Students WHERE student LIKE 'l%'")
    print(mycursor.fetchall())

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
