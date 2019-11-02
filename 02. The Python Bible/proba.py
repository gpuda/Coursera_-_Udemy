board = ["  " for x in range(9)]

def sq_board():
    raw1 = ("|{}|{}|{}|".format(board[0],board[1],board[2]))
    raw2 = ("|{}|{}|{}|".format(board[3],board[4],board[5]))
    raw3 = ("|{}|{}|{}|".format(board[6],board[7],board[8]))

    print()
    print(raw1)
    print(raw2)
    print(raw3)
    print()

def player_move(item):
    if item == "X":
        num = 1
    elif item == "O":
        num = 2
    print ("Player {} on the move!".format(num))

    choice = int(input("Give me a move (1-9)?: "))
    if board[choice -1] == "  ":
        board[choice -1] = item
    else:
        print()
        print("place occupied!")

def victory(item):
    if  (board[0] == item and board[1] == item and board[2] == item) or \
        (board[3] == item and board[4] == item and board[5] == item) or \
        (board[6] == item and board[7] == item and board[8] == item) or \
        (board[0] == item and board[3] == item and board[6] == item) or \
        (board[1] == item and board[4] == item and board[7] == item) or \
        (board[2] == item and board[5] == item and board[8] == item) or \
        (board[0] == item and board[4] == item and board[8] == item) or \
        (board[2] == item and board[4] == item and board[6] == item):
        return True
    else:
        return False

def draw():
    if "  " not in board:
        return True
    else:
        return False

while True:
    sq_board()
    player_move("X")
    if victory("X"):
        print("------ X wins! ------")
        break
    if draw():
        print("------ Draw! ------")
        break
    sq_board()
    player_move("O")
    if victory("O"):
        print("------ O wins! ------")
        break
    if draw():
        print("------ Draw! ------")
        break
