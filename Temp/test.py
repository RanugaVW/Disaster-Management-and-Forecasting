class TuringMachine:
    def __init__(self, tape):
        """
        Initializes the Turing Machine with a given input tape.

        Parameters:
        tape (str): The input string containing 'a', 'b', and 'c' in the required format.

        Attributes:
        self.tape (list): A list representation of the tape with an added blank symbol ('$') at the end.
        self.head (int): Position of the tape head.
        self.state (str): Current state of the Turing Machine.
        self.transitions (dict): Dictionary containing transition rules.
        """
        self.tape = list(tape) + ["$"]  # Add blank symbol at the end
        self.head = 0  # Tape head starts at position 0
        self.state = "Q0"  # Initial state

        # TODO 1: Define all the transitions for the Turing Machine.
        # The transition table follows this format:
        # (Current State, Current Tape Symbol) → (Next State, New Tape Symbol, Move Direction)
        # State Q0 is done for you as an example.
        self.transitions = {
            # State Q0
            ("Q0", "a"): ("Q1", "X", "R"),  
            ("Q0", "Y"): ("Q5", "Y", "R"), 
 
            
            ("Q1", "a"): ("Q1", "a", "R"),   
            ("Q1", "Y"): ("Q1", "Y", "R"),   
            ("Q1", "b"): ("Q2", "Y", "R"),   
 
            
            ("Q2", "b"): ("Q3", "Y", "R"),   
            
            
            ("Q3", "b"): ("Q4", "Y", "L"),   
 
            
            ("Q4", "Y"): ("Q4", "Y", "L"),   
            ("Q4", "b"): ("Q4", "b", "L"),   
            ("Q4", "a"): ("Q4", "a", "L"),   
            ("Q4", "X"): ("Q0", "X", "R"),   
 
        
            ("Q5", "Y"): ("Q5", "Y", "R"),   
            ("Q5", "c"): ("Q6", "c", "R"),   
            ("Q5", "$"): ("ha", "$", "R"),   
 
        
            ("Q6", "c"): ("Q6", "c", "R"),   
            ("Q6", "$"): ("ha", "$", "R")
        }

    def move(self):
        """
        Simulates the Turing Machine execution.

        The machine moves according to its transition rules until it reaches an accepting ("ha")
        or rejecting state.

        Returns:
        str: "Accepted" if the input belongs to the language, "Rejected" otherwise.
        """
        while self.state != "ha":  # ha is the accepting state
            # TODO 2: Read the current symbol at the tape head
            symbol = self.tape[self.head]
            
            
            if (self.state, symbol) in self.transitions:
                # TODO 3: Fetch the next state, replacement symbol, and movement direction
                next_state, new_symbol, direction = self.transitions[(self.state, symbol)]
                
                # TODO 4: Update the tape by replacing the current symbol with the new symbol
                self.tape[self.head] = new_symbol
                
                # TODO 5: Move the tape head in the correct direction
                if direction == "R":
                    self.head += 1
                else:  # direction == "L"
                    self.head -= 1
                
                # TODO 6: Update the turing machine state 
                self.state = next_state
        
            else:
                # TODO 7: If there is no valid transition
                return "Rejected"
        
        # TODO 7: If the machine reaches the accepting state
        return "Accepted"

Input = "abbb"
if any(c in Input for c in ('$', 'X', 'Y')):
    print("Rejected")
else:
    tm = TuringMachine(Input)
    result = tm.move()
    print(result)


