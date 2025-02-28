import os
import urllib.request
import shutil

def download_chess_pieces():
    """
    Download chess piece images from Wikimedia Commons.
    """
    # Create pieces directory if it doesn't exist
    if not os.path.exists('pieces'):
        os.makedirs('pieces')
        print("Created 'pieces' directory.")
    
    # Base URL for chess pieces
    base_url = "https://upload.wikimedia.org/wikipedia/commons/thumb"
    
    # Piece mappings (filename: URL path)
    piece_urls = {
        # White pieces
        "wp.png": "/4/45/Chess_plt45.svg/68px-Chess_plt45.svg.png",  # Pawn
        "wn.png": "/7/70/Chess_nlt45.svg/68px-Chess_nlt45.svg.png",  # Knight
        "wb.png": "/b/b1/Chess_blt45.svg/68px-Chess_blt45.svg.png",  # Bishop
        "wr.png": "/7/72/Chess_rlt45.svg/68px-Chess_rlt45.svg.png",  # Rook
        "wq.png": "/1/15/Chess_qlt45.svg/68px-Chess_qlt45.svg.png",  # Queen
        "wk.png": "/4/42/Chess_klt45.svg/68px-Chess_klt45.svg.png",  # King
        
        # Black pieces
        "bp.png": "/c/c7/Chess_pdt45.svg/68px-Chess_pdt45.svg.png",  # Pawn
        "bn.png": "/e/ef/Chess_ndt45.svg/68px-Chess_ndt45.svg.png",  # Knight
        "bb.png": "/9/98/Chess_bdt45.svg/68px-Chess_bdt45.svg.png",  # Bishop
        "br.png": "/f/ff/Chess_rdt45.svg/68px-Chess_rdt45.svg.png",  # Rook
        "bq.png": "/4/47/Chess_qdt45.svg/68px-Chess_qdt45.svg.png",  # Queen
        "bk.png": "/f/f0/Chess_kdt45.svg/68px-Chess_kdt45.svg.png",  # King
    }
    
    # Download each piece
    for filename, url_path in piece_urls.items():
        full_url = f"{base_url}{url_path}"
        output_path = os.path.join('pieces', filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue
        
        try:
            print(f"Downloading {filename} from {full_url}")
            with urllib.request.urlopen(full_url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    print("Download complete!")

if __name__ == "__main__":
    download_chess_pieces() 