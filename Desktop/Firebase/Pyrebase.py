fimport pyrebase

config = {
    "apiKey": "AIzaSyCJhmjij4ABEZrlIP5581ERp9pXcb6anLk",
    "authDomain": "n-d3a20.firebaseapp.com",
    "databaseURL": "https://n-d3a20.firebaseio.com",
    #"projectId": "n-d3a20",
    "storageBucket": "n-d3a20.appspot.com",
    #"messagingSenderId": "626961674461",
    #"appId": "1:626961674461:web:424708683547daae",
    "serviceAccount": "/Users/michaelpilarski/Desktop/Firebase/n-d3a20-firebase-adminsdk-4gvqc-7e0b540d4f.json"
}

firebase=pyrebase.initialize_app(config)
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("liampilarski2@gmail.com", "Liamlukas=11")
user = auth.refresh(user['refreshToken'])
#print(user)
#print(user['idToken'])
print(user)
db = firebase.database()
data = {"data": {"stuff": "Liam"}, "yeet": {"stuff": "Lukas"}}
db.child("users").set(data)
#path = db.child("user")
#path.child("hey").set(data)
name = db.child("user").child("hey").get().each()
my = (db.child("users").get().each())
for m in my:
    print(m.key())


#print(db.child("me").get().val())
def getNames(location):
    names = []
    data = db.child(location).get().each()
    for d in data:
        names.append(d.key())
    print(names)

getNames("Millburn")
