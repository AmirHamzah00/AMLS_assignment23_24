from A.TaskA import Train_Evaluate_CNN_Model_TaskA, Load_Trained_CNN_Model_TaskA
from B.TaskB import Train_Evaluate_CNN_Model_TaskB
def Model_A():
    selection = True
    while selection:
        print("\nTask A Selection Menu")
        print("\nPlease Select an Option:")
        print("[0] Train and Evalute the CNN Model")
        print("[1] Load and Evaluate Saved CNN Model")
        print("[2] Exit to Main Menu")
        print()
        
        selection = input("Please Enter Key: ") 
        if selection =="0": 
            Train_Evaluate_CNN_Model_TaskA()
            print("\nModel A CNN Trained and Evaluated")
            
        elif selection =="1":
            selection = True
            while selection:
                print("\nPlease Select an Option:")
                print("[0] 1st Saved CNN Model")
                print("[1] 2nd Saved CNN Model")
                print("[2] 3rd Saved CNN Model")
                print("[3] Exit to Previous Selection Menu")
                print()
                
                selection = input("Please Enter Key: ") 
                if selection =="0": 
                    Load_Trained_CNN_Model_TaskA(1)
                    print("\n1st Saved CNN Model Evaluated")
                elif selection =="1":
                    Load_Trained_CNN_Model_TaskA(2)
                    print("\n2nd Saved CNN Model Evaluated")
                elif selection =="2":
                    Load_Trained_CNN_Model_TaskA(3)
                    print("\n3rd Saved CNN Model Evaluated")
                elif selection =="3":
                    print('\nExited!')
                    Model_A()
                else:
                    print("\nInvalid Choice. Re-eneter Key.")
                    
        elif selection =="2":
            print('\nExited!')
            selection_menu()
            
        else:
            print("\nInvalid Choice. Re-eneter Key.") 

def Model_B():
    selection = True
    while selection:
        print("\nTask B Selection Menu")
        print("\nPlease Select an Option:")
        print("[0] Train and Evalute the CNN Model")
        print("[1] Load and Evaluate Saved CNN Model")
        print("[2] Exit")
        print()
        
        selection = input("Please Enter Key: ") 
        if selection =="0": 
            Train_Evaluate_CNN_Model_TaskA()
            print("Model B CNN Trained and Evaluated")
            
        elif selection =="1":
            selection = True
            while selection:
                print("\nPlease Select an Option:")
                print("[0] 1st Saved CNN Model")
                print("[1] 2nd Saved CNN Model")
                print("[2] 3rd Saved CNN Model")
                print("[3] Exit to Previous Selection Menu")
                print()
                
                selection = input("Please Enter Key: ") 
                if selection =="0":
                    print("\n1st Saved CNN Model Evaluated")
                elif selection =="1":
                    print("\n2nd Saved CNN Model Evaluated")
                elif selection =="2":
                    print("\n3rd Saved CNN Model Evaluated")
                elif selection =="3":
                    print('\nExited!')
                    Model_B()
                else:
                    print("\nInvalid Choice. Re-eneter Key.")
            
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