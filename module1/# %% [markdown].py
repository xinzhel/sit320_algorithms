# %% [markdown]
# # Simple Algorithm to play a game of Tic-Tac-Toe

# %%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

# %% [markdown]
# The code is downloaded from here: https://github.com/doctorsmonsters/minimax_tic_tac_toe/blob/main/Minimax%20TicTacToe.ipynb
# I do take any responsibility for its inner working. 
# 
# Your task is to understand it, evaluate it, and make it working. 
# 
# There can be problems in the code. Try fixing them, as part of setting-up the test cases activity
#         

# %% [markdown]
# ## Board Definition

# %%
#define the board
board={1:' ', 2:' ', 3:' ',
       4:' ', 5:' ', 6:' ',
       7:' ', 8:' ', 9:' '}

# %%
#function to print the board
def printBoard(board):
    print(board[1]+'|'+board[2]+'|'+board[3])
    print('-+-+-')
    print(board[4] + '|' + board[5] + '|' + board[6])
    print('-+-+-')
    print(board[7] + '|' + board[8] + '|' + board[9])
    print('\n')

# %%
printBoard(board)

# %%
#function to check is a certain position in the board is empty.
def spaceIsFree(position):
    
    if (board[position]==' '):
        return True
    else:
        return False

# %%
spaceIsFree(1)

# %%
#method to insert letter in space
def insertLetter(letter, position):
    
    if (spaceIsFree(position)):
        board[position]=letter
        printBoard(board)
        
        if (chkDraw()):
            print('Draw!')
        elif (chkForWin()):
            if (letter=='X'):
                print('Bot wins!')
            else:
                print('You win!')
        return
    
    else:
        print('Position taken, please pick a different position.')
        position=int(input('Enter new position: '))
        insertLetter(letter, position)
        
        return

# %%
#function to check if board is draw
def chkDraw():
    
    for key in board.keys():
        if (board[key]==' '):
            return False
    return True

#function to check if one user has won
def chkForWin():
    
    if (board[1]==board[2] and board[1]==board[3] and board[1] !=' '):
        return True
    elif (board[4]==board[5] and board[4]==board[6] and board[4] !=' '):
        return True
    elif (board[7]==board[8] and board[7]==board[9] and board[7] !=' '):
        return True
    elif (board[1]==board[5] and board[1]==board[9] and board[1] !=' '):
        return True
    elif (board[3]==board[5] and board[3]==board[7] and board[3] !=' '):
        return True
    elif (board[1]==board[4] and board[1]==board[7] and board[1] !=' '):
        return True
    elif (board[2]==board[5] and board[2]==board[8] and board[2] !=' '):
        return True
    elif (board[3]==board[6] and board[3]==board[9] and board[3] !=' '):
        return True
    else:
        return False

# %%
#function to check who won
def chkMarkForWin(mark):
    
    if (board[1]==board[2] and board[1]==board[3] and board[1] ==mark):
        return True
    elif (board[4]==board[5] and board[4]==board[6] and board[4] ==mark):
        return True
    elif (board[7]==board[8] and board[7]==board[9] and board[7] ==mark):
        return True
    elif (board[1]==board[5] and board[1]==board[9] and board[1] ==mark):
        return True
    elif (board[3]==board[5] and board[3]==board[7] and board[3] ==mark):
        return True
    elif (board[1]==board[4] and board[1]==board[7] and board[1] ==mark):
        return True
    elif (board[2]==board[5] and board[2]==board[8] and board[2] ==mark):
        return True
    elif (board[3]==board[6] and board[3]==board[9] and board[3] ==mark):
        return True
    else:
        return False

# %%
player = 'O'
bot ='X'

# %%
#function for player move
def playerMove():
    position=int(input('Enter position for O:'))
    insertLetter(player, position)
    return

#function for bot move
def compMove():
    
    bestScore=-1000 #lowest to start with, for comaparison to the score after a certain move
    
    bestMove=0
    
    for key in board.keys(): #for loop to find emplty slots, make a move, calculate the score and see if its higher then best score
        if (board[key]==' '):
            board[key]=bot
            score = minimax(board, 0, False) #calculate the score
            board[key] = ' ' #set board back to what it was
            if (score > bestScore): #compare the score from a certain move to the best score.
                bestScore = score
                bestMove = key
                
    #at the end of the loop, we have the best move figured out.
    insertLetter(bot, bestMove)
    return

# %%
def minimax(board, depth, isMaximizing):
    
    if (chkMarkForWin(bot)):
        return 1
    elif (chkMarkForWin(player)):
        return -1
    elif (chkDraw()):
        return 0
    
    if isMaximizing:
        
        bestScore = -1000
        
        for key in board.keys():
            if board[key]==' ':
                board[key]=bot
                score = minimax(board, 0, False)
                board[key]= ' '
                if (score>bestScore):
                    bestScore = score
        return bestScore
    
    else:
        
        bestScore = 1000
        
        for key in board.keys():
            if board[key]==' ':
                board[key]=player
                score = minimax(board, 0, True)
                board[key]= ' '
                if (score<bestScore):
                    bestScore = score
        return bestScore
    

# %%
while not chkForWin():
    compMove()
    playerMove()

# %%



