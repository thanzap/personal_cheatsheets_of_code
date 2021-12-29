#strings
#Upper/lower a string variable
name="zap"
print(name.upper())
print(name.lower())
print(name[0].upper()) #only the first letter
print(name.islower()) #checks if every letter is lower
print(name.isupper())#checks if every letter is upper
#length of a string
print(len(name)) 
#get the index of a letter in the string
print(name.index("a")) #returns 1
#replace a letter in a string
print(name.replace("a","i"))
#numbers
number=72
print(number + 1) #returns 73
#convert a number to a string
number2=str(number) 
#we can use it to concatenate strings to a print 
print("number is "+number2)
#get the absolute value
print(abs(number))
#get the max 
list=[1,3,5]
print(max(list))
#get the min 
print(min(list))
#get a rounded number
print(round(number))
#get the binary number
print(bin(number))
#import math library
from math import *
from types import CodeType
#get the square root
print(sqrt(number))
#getting user input
name=input('Input your Name: ')
age=input('Input your Age: ')
print("Your name is " + name + " and you are " + age +" years old.")
age=int(input('Input your Age: ')) #convert input to integer but we need to comma , and not + for concatenation
#word replacement program
sentence=input("enter your sentence")
print(sentence)
word1=input("enter word to replace")
word2=input("enter word to replace it with")
print(sentence.replace(word1,word2))
#list
countries=["uk","usa","greece","australia"]
print(countries[0]) #returns uk first item in the list
print(countries[0][1]) #returns u the first letter of the first item
print(countries[1:]) #returns usa greece australia
print(countries[1:3]) #returns only usa greece
print(type(countries))
countries[0]="italy" #change the first value of the list to italy
print(countries[-1]) #returns the last item australia
print(len(countries))
#list can contain both strings  numbers and booleans
countries=["uk",2,"greece",True]
#alternative way to define a list
countries=list(("uk","usa","greece","australia"))
#list methods
list1=[6,3,5,1]
list2=['banana', 'apples', 'oranges','mango']
list1.extend(list2) #returns list1+list2
list2.append("cherry") #adds cherry in the list at the back
list2.insert(1,"cherry") #adds cherry in the 1 index
list2.remove("banana") #removes banana
list2.clear() #delete all values and returns an empty list
list2.index("mango") #returns index of mango
list2.count("mango") #count how many times mango is in the list
list1.sort() #sort in ascending order
list2.reverse() #returns the reverse order
list3=list2.copy() #get a copy of a list
list2.pop(1) #remove the value of this index
del list2[0] #remove the value of this index
del list1 #deletes the whole list
#tuples are lists but they are immutable so we use them if we dont want the values in a tuple to change
three_numbers=(1,3,2)
three_numbers= tuple((1,3,2,"aris"))
#functions in python
def greetings_function(name, age):
    print("welcome " + name + ". You are "+str(age)+" years old.")
greetings_function("zap",26) #the call of a function
# *arguments to pass a variety of values 
def greetings_function(*names):
    print("welcome " +names[0])
greetings_function("John","Tim","Zap") #returns welcome john
#pass values of user input
def greetings_function(name, age):
    print("welcome " + name + ". You are "+str(age)+" years old.")

name= input("Enter your name: ")
age=input("Enter your age: ")
greetings_function(name,age)
#the return keyword
def add_numbers(num1,num2):
    print("hello")
    return num1+num2
#we need to put int otherwise python reads them as strings
num1=int(input("Enter a number"))
num2=int(input("Enter a second number"))
print(add_numbers(num1,num2))
#IF statements
#example
a=2
b=3
if a > b :
    print(str(a) +" is greater than " + str(b))
elif a == b:
    print("They are even")
else:
    print("b is greater than a")
#example2
boy=True
short=True
if boy==True or short==True:
    print("he is a boy or he is short")
#example3
boy=True
short=True
if boy==True and short==True:
    print("he is a boy or he is short")
else:
    print("a is none of the two")
#example4 check if a value of user's input is a string
value=input("enter a value: ")
if type(value) == str:
    print(value+ " is a string")
else:
    print(value + ' is not a string')
#example5 check if a number that user provided can be divided by 5
value=int(input("enter a value: "))
if value % 5 == 0:
    print(value, " can be divided by 5 ")
else:
    print(value, ' cannot be divided by 5')
#example6 check if the length of a sentence is less than 10 
value=input("enter a value: ")
if len(value) < 10:
    print(value + ' is less than 10')
else:
    print(value + " is more than 10")
#example7 check if a number is even or odd
value=int(input("enter a value: "))
if value % 2 == 0:
    print(value, " is an even number")
else:
    print(value, " is an odd number")
#Dictionaries 
my_dict= { 
    "name" : "Zap",
    "age": 26,
    "is_tall" : True,
    "nationality": "african",
    "qualification":"college",
    "friends": ["gregor","bekos"]
}
print(my_dict)
print(my_dict["name"])
#we cannot have 2 keys with the same value no duplicates 
#we can have a key with 2 values
print(len(my_dict))
#While loops
i=1
while i < 6 or i == 6:
    print(i)
    i=i+1 #or i+=1
#for loops 
mylist=["zap","zip","zup"]
for values in mylist:
    print(values)
    if values=="zip":
        break #prints values until zip so it will print zap zip
#example with print after break
mylist=["zap","zip","zup"]
for values in mylist:
    if values=="zip":
        break
    print(values) #prints out only zap 
#example1
mydict={
    "name":"John",
    "age" : 13
}
for values in mydict:
    print(values)
#example with range
for x in range(3,7):
    print(x)
else:
    print("Finished looping")
#2d lists
mylist=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
print(mylist[1][1]) #returns 5
for lists in mylist:
    for row in lists:
        print(row) #returns all the values of mylist 

#building a basic calculator
num1=int(input("Enter a number: "))
num2=int(input("Enter a number: "))
operator=input("enter operation: ")

if operator== "+":
    print("The addition is: ",num1+num2)
elif operator == "-":
    print("The subtraction is: ",num1-num2)
elif operator == "*":
    print("The multiplication is: ",num1*num2)
elif operator == "/":
    print("The division is: ",num1/num2)
else:
    print("Wrong operator")
#Try except in python
try:
    x=int(input("input an integer"))
    print(x)
except: #we can add valueerror,nameerror to be more specific
    print("Value is not an integer")   
else:
    print("Nothing went wrong")
finally:
    print("try except finished")

#Reading files 
country_file=open("C:/Users/user/VSCODE/django/countries.txt","r") #r mean read only, w mean write also, a means to append to the end of file, r+ let us read and write
print(country_file.readable()) #returns true or false is its readable
print(country_file.readlines()) #prints all rows in a list
for rows in country_file.readlines(): #for loop to iterate and print all rows 
    print(rows) 
country_file.close() #closes the file
#Writing files
country_file=open("C:/Users/user/VSCODE/django/countries.txt","w") #if wew dont change the file it overwrites the previous file 
country_file=open("C:/Users/user/VSCODE/django/country.txt","w") #it creates new file country.txt
country_file.write("this is the new country file") #writes this text in it
country_file=open("C:/Users/user/VSCODE/django/countries.txt","a")  #append new text
country_file.write("this is the new line") #append a new text
country_file.write("\nthis is the new line") #append the text in a new line
python_file=open("C:/Users/user/VSCODE/django/new.py","w") #create a new python file
python_file.write("print(\"This is a new file\")") 
#C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe to run a py file
# classes and objects
class myclass:
    x = 5
p1=myclass()
print(p1.x) #prints 5 
#init method
class person:
    def __init__(self, name, age):
        self.name=name
        self.age=age
name=input("enter a name: ")
age=int(input("enter age: "))
p1=person(name,age)
print(p1.name)
print(p1.age)
del p1.age #deletes age of p1
del p1 #deletes the p1 overall
pass #this argument is used when we want to proceed with coding but we dont have finished with the attributes of the class
#example 
def person():
    pass
#Inheritance is getting all atrributes of one class and putting it in a new class
from new import student #example of importing a class from a different file 
class student():
    name="Tim"
    age= "34"
    gender="male"
from new import student
#inheritance of class student
class person(student):
    pass
p1=person()
print(p1.name) #prints out Tim

#Python shell can be used to test any code in python IDLE python

#build a simple login and signup system with Python
print("Create your account")
username=input("Enter a username: ")
password=input("Enter a password: ")
print("You have succesfully created an account \n login now")
usernamelog=input("Login username: ")
passwordlog=input("login password: ")

if username==usernamelog and password==passwordlog:
    print("you have successfully logged in ")
else:
    print("Check out again your credentials")

#Modules and pip in python 
#use pypi to get any module and install it via cmd with
pip install 

#Django
#Installation on cmd
pip install django
mkvirtualenv myapp #creates new virtual environment
django-admin startproject myproject #creates new project
cd myproject #to navigate in project
dir #lists all the files in the directory
deactivate #deactivates virtual; environment
workon myapp #navigates to a virtual environment
lsvirtualenv -b #lists all virtualenv

#django project and django app are different
#django apps are like subsets of the main project
#when we work on a project the root file path must be at project
python manage.py startapp myapp #creates the folder of the app inside the project file
views.py #is the main file 

#Url configuration
myapp->urls.py(new_file) 
#code inside urls.py 
from django.urls import path
from . import views

urlpatterns= [
    path('',views.index,name="index") #empty quotes because its the root url (main page)
]

views.py (to define the views.index method- to define the landing page)
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def index(request):
    return HttpResponse('<h1>aris re </h1>')
#run app
python manage.py runserver #check installation
urls.py (my project) 
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('myapp.urls'))
]
#rerun to see the HTML response 
#templates
create a folder named templates in the root directory
then we need to tell django to search in templates folder
settings.py->installed_apps-> add 'myapp'->templates->dirs:[BASE_DIR/"templates"]
then we create the html file in the templates folder
then we go back to views.py and change the return of index function
to 
return render(request,"index.html")

#sending dynamic data to the template file
views.py->inside the function
def index(request):  
    context={
        'name': 'zap',
        'age' : 26,
        'nationality': 'british'
    }
    return render(request,'index.html',context)

#building a word counter in django
1st we create a new html file inside the templates
2nd we add a new url of the new html file 
3rd then we format the main page to be like this
<h1> Input your text below : </h1>

<form method="" action="counter">
    <textarea name="words" rows="25" cols="100"> </textarea> <br>
    <input type="submit"/>
</form>

action='counter' must be the name of function
name='words' must be the same with the request.get
4th go to views.py and add new function for the counter
def counter(request):
    words=request.GET["words"]
    amount_of_words=len(words.split())
    return render(request,'counter.html',{'amount':amount_of_words})

#GET VS POST 
post is used to encrypt the url to protect sensitive data
get is the default parameter of method in the html form
5th format the url not to be so lengthy
<form method="POST" action="counter">
but it needs CSRF token in order to be executable
{% csrf_token%}
and change also the .GET to .POST in the function views.py
index.html
<h1> Input your text below : </h1>

<form method="POST" action="counter">
    {% csrf_token %}
    <textarea name="words" rows="25" cols="100"> </textarea> <br>
    <input type="submit"/>
</form>
#Static files in Django
static files are the external files we use as templates
we must create a new folder in the root directory for the static files style.css
then go to settings and import os library and go to static field and write this
STATICFILES_DIRS=(os.path.join(BASE_DIR,'static'),) #which is the directory to look on
then go to html file and link the css with this Code
{% load static %}
<link rel='stylesheet' href="{% static 'style.css' %}">

#models is the server side and views is the user sid 
#in models.py we define class of features and we define the data types of each feature
if you have more than 1 feature we can make a for loop in the index.html to generate them
class Feature:
    id:str
    age:int
#admin panel and manipulation of databases
class Feature(models.Model):
    name=models.CharField(max_length=100)
    details=models.CharField(max_length=500)
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

#then we go to the admin.py
from .models import Feature
admin.site.register(Feature)
#views.py function transformation to pull data from the database
def index(request):  
    features=Feature.objects.all()
    return render(request,'index.html',{'features': features})
#index.html 
{% load static %}

 {% for feature in features %}
 <h1> {{feature.name}} </h1>
 <p> {{feature.details}}</p>

 {% endfor %}
#user registration 
new url for registration in urls.py
function in views.py
def register(request):
    return render(request,'register.html')
register.html
<h1>Sign Up below</h1>

<form method='POST' action='register'>
    <p>Username:</p>
    <input type='text' name='username'>
    <p>Email:</p>
    <input type='text' name='E-mail'>
</form>
then go to views.py
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.http import HttpResponse
from .models import Feature
from django.contrib import messages

def register(request):
    if request.method == 'POST':
        username=request.POST["username"]
        email=request.POST["email"]
        password=request.POST["password"]
        password2=request.POST["password2"]

        if password==password2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email already used')
                return redirect('register')
            elif User.objects.filter(username=username).exists():
                messages.info(request,'Username already used')
                return redirect ('register')
            else:
                user=User.objects.create_user(username=username,email=email,password=password)
                user.save();
                return redirect ('login')
        else:
            messages.info(request, 'Password not the same')
            return redirect ('register')
    else:
        return render(request,'register.html')
#final form of the register.html
<h1>Sign Up below</h1>

<style>
h5 {
    color:red;
}  
</style>

{%for message in messages %}
<h5>{{message}}</h5>
{% endfor %}

<form method='POST' action='register'>
    {% csrf_token %}
    <p>Username:</p>
    <input type='text' name='username'/>
    <p>Email:</p>
    <input type='email' name='email'/>
    <p>Password:</p>
    <input type='password' name='password'/>
    <p>Repeat Password:</p>
    <input type='password' name='password2'/> <br>
    <input type="submit" />
</form>

#login 
add new path to urls.py
add new function for login in views.py
def login(request):
    if request.method=="POST":
        username= request.POST["username"]
        password=request.POST["password"]
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect ('/')
        else:
            messages.info(request,"credentials invalid")
            return redirect ('login')
    else:
        return render(request,'login.html')

add new page login.html
<h1> login now
</h1>

<style>
    h3 {
        color:red;
    }  
    </style>
    
    {%for message in messages %}
    <h3>{{message}}</h3>
    {% endfor %}

<form action='login' method='POST'>
    {% csrf_token %}
    <P> Username: </P>
    <input type='text' name='username' />
    <p> Password: </p>
    <input type="password" name="password" /> <br>
    <input type="submit" />

</form>
#logout
go to index.html and add the features
{% load static %}
{%if user.is_authenticated%}
<p> Well Done you are logged in {{user.username}}</p>
<a href="logout">Log Out</a>
{% else %}
<p> Log in for a better experience</p>
<a href="login"> Login </a>
{% endif %}

{% for feature in features %}
<h1> {{feature.name}} </h1>
<p> {{feature.details}}</p>
{% endfor %}

then go to the urls.py and add the new url of the logout
then go to views.py and create the function 

def logout(request):
    auth.logout(request)
    return redirect ('/')
#connect postgreSQL database to the app 
settings.py -> database change mysql3 to postgresql and name to myproject then user postgres then password ' ' host localhost
# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myproject',
        'USER': 'postgres',
        'Password' : ' ',
        'HOST': 'localhost'
    }
}

then go to cmd and pip install psycopg2 and pip install Pillow then python manage.py makemigrations 
then python manage.py migrate then refresh in the postgresql and we can see that is connected

#dynamic url
urls.py in the url 'post/<str:pk>'
then go to views.py and add to the post function def post(request,pk):
def post(request,pk):
    posts=post.object.get(id=pk)
    return render(request,'posts.html',{'posts':posts})


#django REST framework
first install django pip install django
then start a new project and install djangorestframework pip install djangorestframework