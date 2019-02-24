#!/usr/bin/env python3

def print_values(**kwargs):
    for key, value in kwargs.items():
        #print("The value of {} is {}".format(key, value))
        if key == "my_name":
                print(value)

def main():
    args = {
            'my_name':"Sammy", 
            'your_name':"Casey"
        }
    print_values(**args)

if __name__=="__main__":
    main()
