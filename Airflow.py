
#Airflow
Airflow is a platform to program workflows 
-Creation
-Scheduling
-Monitoring
-Can implement programs from os import error, linesep
from any language but workflows are written in Python
-Implements workflows as DAGs which are consisted of the tasks and the dependencies between tasks
-Accessed via code, command-line,web-interface

#DAG code example
# Import the DAG object
from airflow.models import DAG
# Define the default_args dictionary
default_args = {
  'owner': 'dsmith',
  'start_date': datetime(2020, 1, 14),
  'retries': 2
}
# Instantiate the DAG object
etl_dag = DAG('example_etl', default_args=default_args)

#Airflow help
airflow -h
airflow list_dags #To list DAGS
airflow webserver -p 9090 #To start a webserver

#Airflow Operators
-Represent a single task in a workflow
-Run independently
-Generally do not share information
-Various operators to perform different tasks

#BashOperator
from airflow.operators.bash_operator import BashOperator
BashOperator(task_id="bash_example", bash_command='echo "example!"',dag=ml_dag) #in the bash command we can include a .sh file which contains any bash commands

#Task dependency
task1 >> task2 #sets task1 to run before task2

#PythonOperator
from airflow.operators.python_operator import PythonOperator
def printme():
    print("This goes in the logs!")
python_task=PythonOperator(task_id="simple_prit",python_callable=printme,dag=example_dag)

#example
def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, 'wb') as f:
        f.write(r.content)   
    # Use the print method for logging
    print(f"File pulled from {URL} and saved to {savepath}")

from airflow.operators.python_operator import PythonOperator

# Create the task
pull_file_task = PythonOperator(
    task_id='pull_file',
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={'URL':'http://dataserver/sales.json', 'savepath':'latestsales.json'},
    dag=process_sales_dag
)

# Add another Python task
parse_file_task = PythonOperator(
    task_id='parse_file',
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={'inputfile':'latestsales.json', 'outputfile':'parsedfile.json'},
    # Add the DAG
    dag=process_sales_dag
)

#EmailOperator and dependencies
# Import the Operator
from airflow.operators.email_operator import EmailOperator

# Define the task
email_manager_task = EmailOperator(
    task_id='email_manager',
    to='manager@datacamp.com',
    subject='Latest sales JSON',
    html_content='Attached is the latest sales JSON file as requested.',
    files='parsedfile.json',
    dag=process_sales_dag
)
# Set the order of tasks
pull_file_task >> parse_file_task >> email_manager_task


#Airflow scheduling
# Update the scheduling arguments as defined
default_args = {
  'owner': 'Engineering',
  'start_date': datetime(2019, 11, 1),
  'email': ['airflowresults@datacamp.com'],
  'email_on_failure': False,
  'email_on_retry': False,
  'retries': 3,
  'retry_delay': timedelta(minutes=20)
}

dag = DAG('update_dataflows', default_args=default_args, schedule_interval='30 12 * * 3') # 12:30 every wednesday

#Example of an airflow process 
#DAG process_sales
import requests
import json
from datetime import datetime
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator


default_args = {
    'owner':'sales_eng',
    'start_date': datetime(2020, 2, 15),
}

process_sales_dag = DAG(dag_id='process_sales', default_args=default_args, schedule_interval='@monthly')


def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, 'w') as f:
        f.write(r.content)
    print(f"File pulled from {URL} and saved to {savepath}")
    

pull_file_task = PythonOperator(
    task_id='pull_file',
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={'URL':'http://dataserver/sales.json', 'savepath':'latestsales.json'},
    dag=process_sales_dag
)

def parse_file(inputfile, outputfile):
    with open(inputfile) as infile:
      data=json.load(infile)
      with open(outputfile, 'w') as outfile:
        json.dump(data, outfile)
        
parse_file_task = PythonOperator(
    task_id='parse_file',
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={'inputfile':'latestsales.json', 'outputfile':'parsedfile.json'},
    # Add the DAG
    dag=process_sales_dag
)

email_manager_task = EmailOperator(
    task_id='email_manager',
    to='manager@datacamp.com',
    subject='Latest sales JSON',
    html_content='Attached is the latest sales JSON file as requested.',
    files='parsedfile.json',
    dag=process_sales_dag
)

pull_file_task >> parse_file_task >> email_manager_task

#Airflow sensors 
#Filesensor -> checks whether or not a file exists
#ExternalTaskSensor -> wait for a task in another DAG to complete 
#HttpSensor -> Requeest a web URL and check for content
#SqlSensor -> Runs a SQL querry to check for content

#Example
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.contrib.sensors.file_sensor import FileSensor

dag = DAG(
   dag_id = 'update_state',
   default_args={"start_date": "2019-10-01"}
)

precheck = FileSensor(
   task_id='check_for_datafile',
   filepath='salesdata_ready.csv',
   dag=dag)

part1 = BashOperator(
   task_id='generate_random_number',
   bash_command='echo $RANDOM',
   dag=dag
)

import sys
def python_version():
   return sys.version

part2 = PythonOperator(
   task_id='get_python_version',
   python_callable=python_version,
   dag=dag)
   
part3 = SimpleHttpOperator(
   task_id='query_server_for_external_ip',
   endpoint='https://api.ipify.org',
   method='GET',
   dag=dag)
   
precheck >> part3 >> part2

#Airflow executors
#SequentialExecutor -> The default Airflow executor,Runs one task at a time,Useful for debugging
#LocalExecutor -> Runs on a single system, treats tasks as processes, parallelism defined by the user, can utilize all resources
#CeleryExecutor -> Uses a Celery backend as task manager, multiple worker systems can be defined, is significantly more difficult to setup and configure

#Debugging and troubleshooting 
Typical issues
1)DAG wont run on schedule
-Check if scheduler is running
-Fix by running airflow scheduler in command line
-schedule interval hasnt passed
-Not enough tasks free

2)DAG wont load
-DAG not in web UI
-DAG not in airflow list_dags
-Verify the DAG file is in the correct folder
-Determine the DAGS folder via airflow.cfg

3)Syntax errors
-Run airflow list_dags
-Run python3 <dagfile.py>

#SLAs and reporting Airflow
-SLA stands for service level agreement
-The amount of time a task or a DAG should require to run
-SLA Miss is any time the task/DAG doesnt meet the expected timing
-You can view them in Browse: SLA Misses

#Defining SLAs 
- Using the 'sla' argument on the task
task1=BashOperator(task_id="sla_task",
                   bash_command="runcode.sh",
                    sla=timedelta(seconds=30),
                    dag=dag)

- On the default_args dictionary

#Timedelta object
-In the datetime library
from datetime import timedelta
-arguments
timedelta(seconds=30)
timedelta(weeks=2)
timedelta(days=4,hours=10,minutes=20,seconds=30)

#General reporting 
-options for success/failure/error
-Keys in the default_args dictionary
default_args={
   "email" : ["example@gmail.com"],
   "email_on_failure" : True,
   "email_on_retry" : False,
   "email_on_success" : True,
   ....
}
-Within DAGs from the EmailOperator

#Defining an SLA
# Import the timedelta object
from datetime import timedelta

# Create the dictionary entry
default_args = {
  'start_date': datetime(2020, 2, 20),
  'sla': timedelta(minutes=30)
}
# Add to the DAG
test_dag = DAG('test_workflow', default_args=default_args, schedule_interval='@None')

# Adding status emails example
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime

default_args={
    'email': ["airflowalerts@datacamp.com"],
    'email_on_failure': True,
    "email_on_success": True
}
report_dag = DAG(
    dag_id = 'execute_report',
    schedule_interval = "0 0 * * *",
    default_args=default_args
)

precheck = FileSensor(
    task_id='check_for_datafile',
    filepath='salesdata_ready.csv',
    start_date=datetime(2020,2,20),
    mode='reschedule',
    dag=report_dag)

generate_report_task = BashOperator(a
    task_id='generate_report',
    bash_command='generate_report.sh',
    start_date=datetime(2020,2,20),
    dag=report_dag
)

precheck >> generate_report_task

#Working with templates
templated_command="""
   echo "Reading {{ params.filename }}"
"""
t1=BashOperator(task_id="template_task",
    bash_command=templated_command,
    params={"filename" : "file1.txt"}
    dag=example_dag)

#example 
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Create a templated command to execute
# 'bash cleandata.sh datestring'
templated_command = """
bash cleandata.sh {{ ds_nodash }}
"""

# Modify clean_task to use the templated command
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          dag=cleandata_dag)

#More advanced templates JINJA
templated_command="""
{% for filename in params.filenames %}
  echo "Reading {{ filename }}"
{% endfor %}
"""
t1=BashOperator(task_id="template_task",
    bash_command=templated_command,
    params={"filename" : ["file1.txt","file2.txt"]}
    dag=example_dag)

#Variables
Execution Date: {{ ds }}  #YY--MM-DD
Execution Date, no dashes: {{ ds_nodash }} ##YYMMDD
Previous Execution Date: {{ prev_ds }}  #YY--MM-DD
Previous Execution Date, no dashes: {{ prev_ds_nodash }} ##YYMMDD
DAG object: {{ dag }}
Airflow config object: {{ conf }}
Macros: {{ macros }}
{{ macros.datetime }} : The datetime.datetime object 
{{ macros.timedelta }} : The timedelta object 
{{ macros.uuid }} : Python's uuid object

#example using lists with templates
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

filelist = [f'file{x}.txt' for x in range(30)]

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Modify the template to handle multiple files in a 
# single run.
templated_command = """
  <% for filename in params.filenames %>
  bash cleandata.sh {{ ds_nodash }} {{ filename }};
  <% endfor %>
"""

# Modify clean_task to use the templated command
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filenames': filelist},
                          dag=cleandata_dag)


#Sending templated emails
from airflow.models import DAG
from airflow.operators.email_operator import EmailOperator
from datetime import datetime

# Create the string representing the html email content
html_email_str = """
Date: {{ ds }}
Username: {{ params.username }}
"""

email_dag = DAG('template_email_test',
                default_args={'start_date': datetime(2020, 4, 15)},
                schedule_interval='@weekly')
                
email_task = EmailOperator(task_id='email_task',
                           to='testuser@datacamp.com',
                           subject="{{ macros.uuid.uuid4() }}",
                           html_content=html_email_str,
                           params={'username': 'testemailuser'},
                           dag=email_dag)

#Branching
-Provides conditional logic
-Using BranchPythonOperator
from airflow.operators.python_operator import BranchPythonOperator
-Takes a python_callable to return the next task id to follow

#Branching example
def branch_test(**kwargs):
    if int(kwargs["ds_nodash"]) % 2== 0:
        return "even_day_task"
    else:
        return "odd_day_task"

branch_task=BranchPythonOperator(task_id="branch_task",dag=dag,
                                 provide_context=True,
                                 python_callable=branch_test)

start_task >> branch_task >> even_day_task >> even_day_task2
branch_task >> odd_day_task >> odd_day_task2

#Define a BranchPythonOperator
# Create a function to determine if years are different
def year_check(**kwargs):
    current_year = int(kwargs['ds_nodash'][0:4])
    previous_year = int(kwargs['prev_ds_nodash'][0:4])
    if current_year == previous_year:
        return 'current_year_task'
    else:
        return 'new_year_task'

# Define the BranchPythonOperator
branch_task = BranchPythonOperator(task_id='branch_task', dag=branch_dag,
                                   python_callable=year_check, provide_context=True)
# Define the dependencies
branch_dag >> current_year_task
branch_dag >> new_year_task

#Running DAGs and tasks
To run a specific task from command-line
airflow run <dag_id> <task_id> <date>
To run a full DAG 
airflow trigger_dag -e <date> <dag_id>

#Operators reminder 
BashOperator -> Expects a bash_command
PythonOperator -> expects a python_callable
BranchPythonOperator -> requires a python_callable and provide_context=True. The callable must accept **kwargs.
FileSensor- requires filepath argument and might need mode or poke_interval attributes

#Template reminders
Many objects in Airflow can use templates
Certain fields may use templated strings, while others do not
One way to check is to use built-in documentation:
1) Open python3 interpreter 
2) Import necessary libraries 
3) At prompt, run help(<Airflow object>), ie ,help(BashOperator)
4) Look for a line that referencing template_fields. This will specify any of the arguments that can use templates. 

#Creates a file in the following path
touch /home/repl/workspace/startprocess.txt

#Production pipeline 
from airflow.models import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from dags.process import process_data
from datetime import datetime, timedelta

# Update the default arguments and apply them to the DAG.

default_args = {
  'start_date': datetime(2019,1,1),
  'sla': timedelta(minutes=90)
}
    
dag = DAG(dag_id='etl_update', default_args=default_args)

sensor = FileSensor(task_id='sense_file', 
                    filepath='/home/repl/workspace/startprocess.txt',
                    poke_interval=45,
                    dag=dag)

bash_task = BashOperator(task_id='cleanup_tempfiles', 
                         bash_command='rm -f /home/repl/*.tmp',
                         dag=dag)

python_task = PythonOperator(task_id='run_processing', 
                             python_callable=process_data,
                             provide_context=True,
                             dag=dag)


email_subject="""
  Email report for {{ params.department }} on {{ ds_nodash }}
"""


email_report_task = EmailOperator(task_id='email_report_task',
                                  to='sales@mycompany.com',
                                  subject=email_subject,
                                  html_content='',
                                  params={"department": 'Data subscription services'},
                                  dag=dag)


no_email_task = DummyOperator(task_id='no_email_task', dag=dag)


def check_weekend(**kwargs):
    dt = datetime.strptime(kwargs['execution_date'],"%Y-%m-%d")
    # If dt.weekday() is 0-4, it's Monday - Friday. If 5 or 6, it's Sat / Sun.
    if (dt.weekday() < 5):
        return "email_report_task"
    else:
        return "no_email_task"
    
    
branch_task = BranchPythonOperator(task_id='check_if_weekend',
                                   python_callable=check_weekend,
                                   provide_context=True,
                                   dag=dag)

    
sensor >> bash_task >> python_task

python_task >> branch_task >> [email_report_task, no_email_task]











