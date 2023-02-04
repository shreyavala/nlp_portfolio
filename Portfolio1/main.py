# Name: Shreya Valaboju
# Course/Section: CS 4395.001
# Notes: Refer to file named "overview_portfolio1.txt" on instructions on how to run, extra notes, and analysis
# Github Link: https://github.com/shreyavala/nlp_portfolio

# import necessary libraries
import sys
import pathlib
import re
import pickle


"""
Person class that creates Employee objects. Holds information about each employee
...

Attributes
----------
first : str
    employee first name
last : str
    employee last name
mi : str
    employee middle initial, if available
id : str
    employee id
phone : str
    employee office phone number

Methods
-------
displau()
    prints information about each employee/Person object
    """
class Person:
    def __init__(self, first, last, mi, id, phone):
        self.first = first
        self.last = last
        self.mi = mi
        self.id = id
        self.phone = phone


    def display(self): #display/print information about each employee
        print('\nEmployee ID: ', self.id)
        print(self.first+  " "+ self.mi + ". "+ self.last)
        print(self.phone)

# check office phone using regex (in form of '123-456-7890')
def checkOfficePhone(phone_number):
    if re.match('^\d{3}-\d{3}-\d{4}$', phone_number):
        return True
    else:
        return False

# check id using regex (2 letters followed by 4 digits)
def checkID(emp_id):
    if re.match('^[a-zA-Z]{2}\d{4}$', emp_id):
        return True
    else:
        return False


def processEmployee(text):
    '''
    reads in each employee's information line by line, and preprocesses format of data.
            Parameters:
                    text (str): string of all the data from the input/data csv file
            Returns:
                    returns a dictionary with person(employees) objects
    '''

    employee_dict ={} #holds a dictionary of employees (Person objects_
    lines_list = text.splitlines() #read line by line
    for line in lines_list[1:]: #skip header line(assumes there is header)
        # split by comma
        employee_info = line.split(",")

        # capital case names
        employee_info[0] = employee_info[0].upper()
        employee_info[1] = employee_info[1].upper()

        # change empty middle intial to 'X'
        if not employee_info[2]:
            employee_info[2] = 'X'
        else:
            employee_info[2] = employee_info[2].upper()

        # check phone using regex (3 digits - 3 digits - 4 digits)
        while checkOfficePhone(employee_info[4]) is False:
            print("\nPhone '" + employee_info[4] + "' is invalid")
            # take user input to fix phone number
            employee_info[4] = input("Enter phone number in form 123-456-7890:  ")

        # check id using regex (2 letters followed by 4 digits)
        id_exists = bool(employee_dict) and (employee_info[3] in employee_dict) #flag to check for duplicate keys
        while checkID(employee_info[3]) is False or id_exists:
            if id_exists:
                print("ERROR: Duplicate ID")    #print error message if id already exists
            print("ID is invalid: " + employee_info[3])
            print("ID is two letters followed by four digits")  # take user input to fix id
            employee_info[3] = input("Please enter a valid ID:  ")
            employee_info[3]= employee_info[3] .upper()
            id_exists = bool(employee_dict) and (employee_info[3] in employee_dict)

        # create employee object once all employee fields are standardized and valid
        employee = Person(employee_info[0],employee_info[1],employee_info[2], employee_info[3],employee_info[4])

        # place in a dictionary (key = ID)
        employee_dict[employee_info[3]] = employee
        #print(employee_info)

    return employee_dict




if __name__ == '__main__':

    if len(sys.argv) < 2:   #check if number of arguments is atleast 2, if not terminate program
        print("ERROR: Please enter argument (sysarg) containing input/data file relative path. Re-run program.")
        quit()

    try:
        with open(pathlib.Path.cwd().joinpath(sys.argv[1]), 'r') as f: # find data file
            text_in = f.read()
    except FileNotFoundError:
        print("ERROR: Input/data file provided cannot be found. Please re-run program.")
        quit()

    # save each employee into a dictionary
    emp_dict= processEmployee(text_in)
    #print(emp_dict)

    # save entire dictionary into pickle file
    pickle.dump(emp_dict,open('emp_dict_file.p','wb'))

    # open pickle file, print using display()
    emp_dict_in = pickle.load(open('emp_dict_file.p','rb'))
    # iterate through each key value pair, call display on each object(employee) and print
    print('\nEmployee List: ')
    for val in emp_dict_in.values():
        val.display()

    #print(emp_dict_in)

