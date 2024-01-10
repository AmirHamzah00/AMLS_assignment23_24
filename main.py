from A.TaskA import CNN_ModelA

def Model_A():
    print("Model A CNN Exceuted")
    CNN_ModelA()

def Model_B():
    print("Model B CNN Exceuted")

selection = True
while selection:
    
    print("\nAMLS Assignment 23/24\n")
    print("Please Select Which Task To Execute:")
    print("[0] Execute Task A")
    print("[1] Execute Task B")
    print("[2] Exit")
    print()
    
    selection = input("Please Enter Key: ") 
    if selection =="0": 
        Model_A() 
    elif selection =="1":
        Model_B() 
    elif selection =="2":
        print('\nMenu Exited!\n')
        exit()
    else:
      print("\nInvalid Choice. Re-eneter Key.") 