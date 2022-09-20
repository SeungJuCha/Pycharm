class CreditCard_in_NB:
    """A consumer credit card."""

    def __init__(self, customer, bank, acnt, limit,color):
        """Create a new credit card instance.

        The initial balance is zero.

        customer  the name of the customer (e.g., 'John Bowman')
        bank      the name of the bank (e.g., 'California Savings')
        acnt      the acount identifier (e.g., '5391 0375 9387 5309')
        limit     credit limit (measured in dollars)
        """
        self._customer = customer
        self._bank = bank
        self._account = acnt
        self._limit = limit
        self._balance = 0
        self._color = color


    def get_customer(self):
        """Return name of the customer."""
        return self._customer

    def get_bank(self):
        """Return the bank's name."""
        return self._bank

    def get_account(self):
        """Return the card identifying number (typically stored as a string)."""
        return self._account

    def get_limit(self):
        """Return current credit limit."""
        return self._limit

    def get_balance(self):
        """Return current balance."""
        return self._balance

    def get_color_of_credit_card(self):
        """Return current color."""
        return self._color

    def set_color_of_credit_card(self,color):
        self._color = color


    def charge(self, price):
        """Charge given price to the card, assuming sufficient credit limit.

        Return True if charge was processed; False if charge was denied.
        """
        if price + self._balance > self._limit:  # if charge would exceed limit,
            return False  # cannot accept charge
        else:
            self._balance += price
            return True

    def make_payment(self, amount):
        """Process customer payment that reduces balance."""
        self._balance -= amount

    def print_info(self):
        print('Customer =', self.get_customer())
        print('Bank =', self.get_bank())
        print('Account =', self.get_account())
        print('Limit =', self.get_limit())
        print('Balance =', self.get_balance())
        print('Color =' , self.get_color_of_credit_card())

    class Point2D:
        def __init__(self, x, y):
            self.__x = x
            self.__y = y

        def get_X(self):
            return self.__x

        def set_X(self, x):
            self.__x = x

        def get_Y(self):
            return self.__y

        def set_Y(self, y):
            self.__y = y


# def select_color_of_credit_card(Select_color):
#     """Select the color"""
#     Select_color = input('Please select your color = ')
#     return Select_color

