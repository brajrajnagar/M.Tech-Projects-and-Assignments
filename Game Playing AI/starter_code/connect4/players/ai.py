import random
import numpy as np
from typing import List, Tuple, Dict
import time as tmlib
from connect4.utils import get_pts, get_valid_actions, Integer


class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        # Do the rest of your implementation here

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        # raise NotImplementedError('Whoops I don\'t know what to do')
        begin = tmlib.time()

        board_org, p_out_org = state
        board = board_org.copy()
        n_row, n_col = board.shape
        p_out = {1:p_out_org[1].get_int(), 2:p_out_org[2].get_int()}

        ai_num = self.player_number
        if ai_num == 1:
            opp_num = 2
        else:
            opp_num = 1

        window_spec =  [4,3,2]
        sunya = 0

        def get_allowed_actions(player_num, new_state):
            valid_actions = []
            board = new_state[0]
            pop_left = new_state[1][player_num]
            n = board.shape[1]
            for col in range(n):
                if 0 in board[:,col]:
                    valid_actions.append((col, False))

                if pop_left > 0:
                    for col in range(n):
                        if col % 2 == player_num - 1:
                            if board[:, col].any():
                                valid_actions.append((col, True))
            return valid_actions



        def terminal_node(valid_actions):
            if len(valid_actions) == 0:
                return True
            else:
                return False

        def updated_board(board, p_out, move, player_num):
            new_board = board.copy()
            new_p_out = p_out.copy()
            r = new_board.shape[0]
            col, pop_check = move
            if pop_check == False:
                row = 0
                for i in range(r-1):
                    if new_board[(r-1)-i][col] == 0:
                        row = (r-1)-i
                        break
                    else:
                        continue
                new_board[row][col] = player_num
            else:
                for i in range(r-1):
                    new_board[(r-1)-i][col] = new_board[(r-1)-(i+1)][col]
                new_board[0][col] = 0
                new_p_out[player_num] -= 1  
            return new_board, new_p_out


        def win_score(window_, win_length):
            score = 0
            ai_pieces = window_.count(ai_num)
            opp_pieces = window_.count(opp_num)
            n_zeros = window_.count(sunya)
            

            if ai_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score += 12.5

            elif opp_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score -= 18.75    #12.5*1.5

            elif ai_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score += 5
                elif win_length == 3 and n_zeros == 1:
                    score += 6.66

            elif opp_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score -= 7.5   #5*1.5
                elif win_length == 3 and n_zeros == 1:
                    score -= 9.99    #6.66*1.5

            return score


        def win_score_col(window_, win_length):
            score = 0
            ai_pieces = window_.count(ai_num)
            opp_pieces = window_.count(opp_num)
            n_zeros = window_.count(sunya)
            

            if ai_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score += 15.625    #12.5*1.25

            elif opp_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score -= 23.4375    #12.5*1.5*1.25

            elif ai_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score += 6.25    #5*1.25
                elif win_length == 3 and n_zeros == 1:
                    score += 8.325    #6.66*1.25

            elif opp_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score -= 9.375    #5*1.5*1.25
                elif win_length == 3 and n_zeros == 1:
                    score -= 12.4875    #6.66*1.5*1.25

            return score
        
        win_pt = [0,0,10,25,100]   

        def get_score(player_number, row):
            score = 0
            n = len(row)
            j = 0
            while j < n:
                if row[j] == player_number:
                    count = 0
                    while j < n and row[j] == player_number:
                        count += 1
                        j += 1
                    k = len(win_pt) - 1
                    score += win_pt[count % k] + (count // k) * win_pt[k]
                else:
                    j += 1
            return score

        # def cluster_score(board):
        #     score = 0
        #     if board[0][0]==opp_num and board[0][1]==opp_num and board[1][0]==opp_num and board[1][1]==opp_num:
        #         score -= 100
        #         return score
        #     else:
        #         return score


        def win_pos(board):
            score = 0

            center_col = [int(i) for i in list(board[:, n_col//2])]
            center_count = center_col.count(ai_num)
            score += center_count*3

            center_col = [int(i) for i in list(board[:, n_col//2])]
            center_count = center_col.count(opp_num)
            score += -center_count*3

            # for i in range(n_row-1):
            #     for j in range(n_col-1):
            #         sq = np.array([[board[i][j],board[i][j+1]],
            #                         [board[i+1][j],board[i+1][j+1]]])
            #         score += cluster_score(sq)

            for win_length in window_spec:
                
                for r in range(n_row):
                    row_array = [int(i) for i in list(board[r,:])]
                    score += (get_score(ai_num,row_array) - get_score(opp_num,row_array)*1.5)
                    c = 0
                    while c < (n_col-(win_length-1)):
                        window_ = row_array[c:c+win_length]
                        window_score = win_score(window_, win_length)
                        score += window_score
                        c += 1

                for c2 in range(n_col):
                    col_array = [int(i) for i in list(board[:,c2])]
                    score += (get_score(ai_num,col_array) - get_score(opp_num,col_array)*1.5)
                    r2 = 0
                    while r2 < (n_row-(win_length-1)):
                        window_ = col_array[r2:r2+win_length]
                        window_score = win_score_col(window_, win_length)
                        score += window_score
                        r2 += 1


            def diagonals_primary(board):
                m, n = board.shape
                for k in range(n + m - 1):
                    diag = []
                    for j in range(max(0, k - m + 1), min(n, k + 1)):
                        i = k - j
                        diag.append(board[i, j])
                    yield diag

            def diagonals_secondary(board: np.array) -> List[int]:
                m, n = board.shape
                for k in range(n + m - 1):
                    diag = []
                    for x in range(max(0, k - m + 1), min(n, k + 1)):
                        j = n - 1 - x
                        i = k - x
                        diag.append(board[i][j])
                    yield diag


            for diag in diagonals_primary(board):
                score += (get_score(ai_num, diag) - get_score(opp_num, diag)*1.5)
            for diag in diagonals_secondary(board):
                score += (get_score(ai_num, diag) - get_score(opp_num, diag)*1.5)

            return score


        
        def minimax(board, p_out, depth, alpha, beta, player_num):
            new_state = (board, p_out)
            valid_actions = get_allowed_actions(player_num, new_state)
            if depth==0 or terminal_node(valid_actions):
                if (tmlib.time() - begin) < self.time-1.5:
                    return None, win_pos(board)
                else:
                    return None, None
                    

            elif player_num == ai_num:
                value = -1e20
                best_move = random.choice(valid_actions)
                for move in valid_actions:
                    child_board, child_p_out = updated_board(board ,p_out ,move, player_num)
                    points = minimax(child_board, child_p_out, depth-1, alpha, beta, opp_num)[1]
                    # print(points)
                    if points == None:
                        return (None,None), None
                    elif points > value:
                        value = points
                        best_move = move

                    if value >= beta:
                        break
                    alpha = max(alpha, value)
                return best_move, value
            
            else:
                value = 1e20
                best_move = random.choice(valid_actions)
                for move in valid_actions:
                    child_board, child_p_out = updated_board(board ,p_out ,move, player_num)
                    point = minimax(child_board, child_p_out, depth-1, alpha, beta, ai_num)[1]
                    if point == None:
                        return (None,None), None
                    elif point < value:
                        value = point
                        best_move = move

                    if value <= alpha:
                        break
                    beta = min(beta, value)
                return best_move, value


        valid_act = get_allowed_actions(ai_num, (board, p_out))
        action, is_popout = random.choice(valid_act)
        depth = 1
        while True:
            act,popout = minimax(board, p_out, depth, -1e20, 1e20, ai_num)[0]
            if act == None:
                break
            elif depth>20 and act == action and popout == is_popout:
                break
            else:
                action = act
                is_popout = popout
                # print(depth)
                depth += 1

        return action, is_popout














    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        # raise NotImplementedError('Whoops I don\'t know what to do')


        begin = tmlib.time()

        board_org, p_out_org = state
        board = board_org.copy()
        n_row, n_col = board.shape
        p_out = {1:p_out_org[1].get_int(), 2:p_out_org[2].get_int()}

        ai_num = self.player_number
        if ai_num == 1:
            opp_num = 2
        else:
            opp_num = 1

        window_spec =  [4,3,2]
        sunya = 0

        def get_allowed_actions(player_num, new_state):
            valid_actions = []
            board = new_state[0]
            pop_left = new_state[1][player_num]
            n = board.shape[1]
            for col in range(n):
                if 0 in board[:,col]:
                    valid_actions.append((col, False))

                if pop_left > 0:
                    for col in range(n):
                        if col % 2 == player_num - 1:
                            if board[:, col].any():
                                valid_actions.append((col, True))
            return valid_actions



        def terminal_node(valid_actions):
            if len(valid_actions) == 0:
                return True
            else:
                return False

        def updated_board(board, p_out, move, player_num):
            new_board = board.copy()
            new_p_out = p_out.copy()
            r = new_board.shape[0]
            col, pop_check = move
            if pop_check == False:
                row = 0
                for i in range(r-1):
                    if new_board[(r-1)-i][col] == 0:
                        row = (r-1)-i
                        break
                    else:
                        continue
                new_board[row][col] = player_num
            else:
                for i in range(r-1):
                    new_board[(r-1)-i][col] = new_board[(r-1)-(i+1)][col]
                new_board[0][col] = 0
                new_p_out[player_num] -= 1  
            return new_board, new_p_out


        def win_score(window_, win_length):
            score = 0
            ai_pieces = window_.count(ai_num)
            opp_pieces = window_.count(opp_num)
            n_zeros = window_.count(sunya)
            

            if ai_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score += 12.5

            elif opp_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score -= 18.75    #12.5*1.5

            elif ai_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score += 5
                elif win_length == 3 and n_zeros == 1:
                    score += 6.66

            elif opp_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score -= 7.5   #5*1.5
                elif win_length == 3 and n_zeros == 1:
                    score -= 9.99    #6.66*1.5

            return score


        def win_score_col(window_, win_length):
            score = 0
            ai_pieces = window_.count(ai_num)
            opp_pieces = window_.count(opp_num)
            n_zeros = window_.count(sunya)
            

            if ai_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score += 15.625    #12.5*1.25

            elif opp_pieces == 3:
                if win_length == 4 and n_zeros == 1:
                    score -= 23.4375    #12.5*1.5*1.25

            elif ai_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score += 6.25    #5*1.25
                elif win_length == 3 and n_zeros == 1:
                    score += 8.325    #6.66*1.25

            elif opp_pieces == 2:
                if win_length == 4 and n_zeros == 2:
                    score -= 9.375    #5*1.5*1.25
                elif win_length == 3 and n_zeros == 1:
                    score -= 12.4875    #6.66*1.5*1.25

            return score
        
        win_pt = [0,0,10,25,100]   

        def get_score(player_number, row):
            score = 0
            n = len(row)
            j = 0
            while j < n:
                if row[j] == player_number:
                    count = 0
                    while j < n and row[j] == player_number:
                        count += 1
                        j += 1
                    k = len(win_pt) - 1
                    score += win_pt[count % k] + (count // k) * win_pt[k]
                else:
                    j += 1
            return score

        # def cluster_score(board):
        #     score = 0
        #     if board[0][0]==opp_num and board[0][1]==opp_num and board[1][0]==opp_num and board[1][1]==opp_num:
        #         score -= 100
        #         return score
        #     else:
        #         return score


        def win_pos(board):
            score = 0

            center_col = [int(i) for i in list(board[:, n_col//2])]
            center_count = center_col.count(ai_num)
            score += center_count*3

            center_col = [int(i) for i in list(board[:, n_col//2])]
            center_count = center_col.count(opp_num)
            score += -center_count*3

            # for i in range(n_row-1):
            #     for j in range(n_col-1):
            #         sq = np.array([[board[i][j],board[i][j+1]],
            #                         [board[i+1][j],board[i+1][j+1]]])
            #         score += cluster_score(sq)

            for win_length in window_spec:
                
                for r in range(n_row):
                    row_array = [int(i) for i in list(board[r,:])]
                    score += (get_score(ai_num,row_array) - get_score(opp_num,row_array)*1.5)
                    c = 0
                    while c < (n_col-(win_length-1)):
                        window_ = row_array[c:c+win_length]
                        window_score = win_score(window_, win_length)
                        score += window_score
                        c += 1

                for c2 in range(n_col):
                    col_array = [int(i) for i in list(board[:,c2])]
                    score += (get_score(ai_num,col_array) - get_score(opp_num,col_array)*1.5)
                    r2 = 0
                    while r2 < (n_row-(win_length-1)):
                        window_ = col_array[r2:r2+win_length]
                        window_score = win_score_col(window_, win_length)
                        score += window_score
                        r2 += 1


            def diagonals_primary(board):
                m, n = board.shape
                for k in range(n + m - 1):
                    diag = []
                    for j in range(max(0, k - m + 1), min(n, k + 1)):
                        i = k - j
                        diag.append(board[i, j])
                    yield diag

            def diagonals_secondary(board: np.array) -> List[int]:
                m, n = board.shape
                for k in range(n + m - 1):
                    diag = []
                    for x in range(max(0, k - m + 1), min(n, k + 1)):
                        j = n - 1 - x
                        i = k - x
                        diag.append(board[i][j])
                    yield diag


            for diag in diagonals_primary(board):
                score += (get_score(ai_num, diag) - get_score(opp_num, diag)*1.5)
            for diag in diagonals_secondary(board):
                score += (get_score(ai_num, diag) - get_score(opp_num, diag)*1.5)

            return score


        
        def expectimax(board, p_out, depth, player_num):
            new_state = (board, p_out)
            valid_actions = get_allowed_actions(player_num, new_state)
            if depth==0 or terminal_node(valid_actions):
                if (tmlib.time() - begin) < self.time-1.5:
                    return None, win_pos(board)
                else:
                    return None, None

            elif player_num == ai_num:
                value = -1e20
                best_move = random.choice(valid_actions)
                for move in valid_actions:
                    child_board, child_p_out = updated_board(board ,p_out ,move, player_num)
                    points = expectimax(child_board, child_p_out, depth-1, opp_num)[1]
                    if points == None:
                        return (None,None), None
                    elif points > value:
                        value = points
                        best_move = move
                return best_move, value
            
            else:
                sum_min = 0
                best_move = random.choice(valid_actions)
                for move in valid_actions:
                    child_board, child_p_out = updated_board(board ,p_out ,move, player_num)
                    point = expectimax(child_board, child_p_out, depth-1, ai_num)[1]
                    if point == None:
                        return (None,None), None
                    else:
                        sum_min += point
                return  best_move, sum_min/len(valid_actions)


        valid_act = get_allowed_actions(ai_num, (board, p_out))
        action, is_popout = random.choice(valid_act)
        depth = 1
        while True:
            act,popout = expectimax(board, p_out, depth, ai_num)[0]
            if act == None:
                break
            elif depth>20 and act==action and popout == is_popout:
                break
            else:
                action = act
                is_popout = popout
                # print(depth)
                depth += 1

        return action, is_popout