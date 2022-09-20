
from CreditCard import CreditCard_in_NB

Name = input('Please write your name = ')
Bank = input('What is your Bank = ')
Account = input('What is your account = ')
Limit = input('Please select your limit =')
Color = input('Choose your color = ')

wallet = []
wallet.append(CreditCard_in_NB(Name,Bank, Account,Limit,Color))
wallet.append(CreditCard_in_NB('Seunghwan Choi', 'Woori Bank','3485 0399 3395 1954',2500,'green'))
wallet.append(CreditCard_in_NB('Seunghwan Choi', 'KB Bank','5391 0375 9387 5309',3000,color =  'blue'))

for c in range(3):
  wallet[c].print_info()

