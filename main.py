def Model_A():
    selection = True
    while selection:
        print("\nPlease Select an Option:")
        print("[0] Train and Evalute the CNN Model")
        print("[1] Load and Evaluate Saved CNN Model")
        print("[2] Exit to Main Menu")
        print()
        
        selection = input("Please Enter Key: ") 
        if selection =="0": 
            print("\nModel A CNN Trained and Evaluated")
        elif selection =="1":
            print("\nSaved Model A CNN Evaluated")
        elif selection =="2":
            print('\nExited!')
            selection_menu()
        else:
            print("\nInvalid Choice. Re-eneter Key.") 

def Model_B():
    selection = True
    while selection:
        print("\nPlease Select an Option:")
        print("[0] Train and Evalute the CNN Model")
        print("[1] Load and Evaluate Saved CNN Model")
        print("[2] Exit")
        print()
        
        selection = input("Please Enter Key: ") 
        if selection =="0": 
            print("Model A CNN Trained and Evaluated")
        elif selection =="1":
            print("Saved Model A CNN Evaluated")
        elif selection =="2":
            print('\nExited!')
            selection_menu()
        else:
            print("\nInvalid Choice. Re-eneter Key.")
        
def selection_menu():
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

selection_menu()