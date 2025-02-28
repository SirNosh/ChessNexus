from chess_engine import ChessEngine

def print_instructions():
    """Print the game instructions"""
    print("\nChess Game - Command Line Interface")
    print("-----------------------------------")
    print("Enter moves in algebraic notation (e.g., 'e2e4' to move from e2 to e4)")
    print("Commands:")
    print("  'quit' - Exit the game")
    print("  'reset' - Reset the board")
    print("  'undo' - Undo the last move")
    print("  'help' - Show these instructions")
    print("-----------------------------------\n")

def algebraic_to_index(notation):
    """Convert algebraic notation (e.g., 'e4') to board indices (row, col)"""
    if len(notation) != 2:
        return None
    
    col = ord(notation[0].lower()) - ord('a')
    row = 8 - int(notation[1])
    
    if 0 <= row < 8 and 0 <= col < 8:
        return (row, col)
    return None

def index_to_algebraic(row, col):
    """Convert board indices to algebraic notation"""
    return chr(col + ord('a')) + str(8 - row)

def parse_move(move_str):
    """Parse a move string (e.g., 'e2e4') into start and end positions"""
    if len(move_str) != 4:
        return None, None
    
    start = algebraic_to_index(move_str[:2])
    end = algebraic_to_index(move_str[2:])
    
    if start is None or end is None:
        return None, None
    
    return start, end

def main():
    """Main function to run the chess game in command line"""
    game = ChessEngine()
    print_instructions()
    
    while True:
        # Print the current board
        game.print_board()
        
        # Print game status
        if game.checkmate:
            winner = "Black" if game.white_to_move else "White"
            print(f"\nCheckmate! {winner} wins!")
            break
        elif game.stalemate:
            print("\nStalemate! The game is a draw.")
            break
        
        # Get player input
        current_player = "White" if game.white_to_move else "Black"
        move_input = input(f"\n{current_player} to move: ").strip().lower()
        
        # Process commands
        if move_input == 'quit':
            print("Thanks for playing!")
            break
        elif move_input == 'reset':
            game.reset_board()
            print("Game reset!")
            continue
        elif move_input == 'undo':
            if len(game.move_log) > 0:
                game.undo_move()
                print("Move undone!")
            else:
                print("No moves to undo!")
            continue
        elif move_input == 'help':
            print_instructions()
            continue
        
        # Parse the move
        start, end = parse_move(move_input)
        if start is None or end is None:
            print("Invalid move format! Use format 'e2e4'.")
            continue
        
        # Check if the move is valid
        valid_moves = game.get_valid_moves()
        move_is_valid = False
        
        for move in valid_moves:
            if move[0] == start and move[1] == end:
                move_is_valid = True
                break
        
        if move_is_valid:
            game.make_move(start, end)
            print(f"Moved from {move_input[:2]} to {move_input[2:4]}")
        else:
            piece = game.board[start[0]][start[1]]
            if piece == 0:
                print("There is no piece at that position!")
            elif (game.white_to_move and piece < 0) or (not game.white_to_move and piece > 0):
                print("That's not your piece!")
            else:
                print("Invalid move! Try again.")

if __name__ == "__main__":
    main() 