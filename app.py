import streamlit as st
import chess
import chess.svg
import torch
import torch.nn as nn
import numpy as np
import random
import base64
import os

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.relu(x + residual)
        return x

# MODEL 1: Bobby Fischer Engine (Compact, Single Head)
class FischerNet(nn.Module):
    def __init__(self, input_channels=19, num_blocks=4):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 73, kernel_size=1, bias=False), 
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)
        return self.policy_head(x)

# MODEL 2: World Elite Masters Engine (Wide Trunk, Dual Head)
class DualHeadChessNet(nn.Module):
    def __init__(self, input_channels=19, num_blocks=6): 
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 73, kernel_size=1, bias=False), 
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

# ==========================================
# 2. COORDINATE TRANSFORMATIONS & PARSERS
# ==========================================

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    piece_map = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    tensor = torch.zeros(19, 8, 8, dtype=torch.float32)
    for sq, p in board.piece_map().items():
        row, col = divmod(sq, 8)
        ch = piece_map[p.piece_type] if p.color == chess.WHITE else 6+piece_map[p.piece_type]
        tensor[ch, row, col] = 1.0
    if board.turn == chess.WHITE: tensor[12,:,:] = 1.0
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[13, r, c] = 1.0
    cr = board.castling_rights
    tensor[14,:,:] = 1.0 if cr & chess.BB_H1 else 0.0
    tensor[15,:,:] = 1.0 if cr & chess.BB_A1 else 0.0
    tensor[16,:,:] = 1.0 if cr & chess.BB_H8 else 0.0
    tensor[17,:,:] = 1.0 if cr & chess.BB_A8 else 0.0
    tensor[18,:,:] = board.halfmove_clock / 100.0
    return tensor

def move_to_policy_index(move: chess.Move):
    fr, fc = divmod(move.from_square, 8)
    tr, tc = divmod(move.to_square, 8)
    dr, dc = tr - fr, tc - fc
    knight_offsets = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
    if (abs(dr), abs(dc)) in [(1,2), (2,1)]:
        try: return 56 + knight_offsets.index((dr, dc)), tr, tc
        except ValueError: return None
    dirs = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    for dir_idx, (ddr, ddc) in enumerate(dirs):
        if (ddr == 0 and ddc == 0) or (dr == 0 and dc == 0): continue
        if ddr != 0 and ddc != 0:
            if abs(dr) != abs(dc): continue
            if dr//abs(dr) != ddr or dc//abs(dc) != ddc: continue
            dist = abs(dr)
        elif ddr == 0:
            if dr != 0: continue
            if dc//abs(dc) != ddc: continue
            dist = abs(dc)
        else:
            if dc != 0: continue
            if dr//abs(dr) != ddr: continue
            dist = abs(dr)
        if 1 <= dist <= 7: return dir_idx * 7 + (dist - 1), tr, tc
    if move.promotion and move.promotion != chess.QUEEN:
        if tr == 7: forward_dr, left_dc, right_dc = -1, -1, 1
        elif tr == 0: forward_dr, left_dc, right_dc = 1, 1, -1
        else: return None
        if dr == forward_dr and dc == 0: dir_idx = 1
        elif dr == forward_dr and dc == left_dc: dir_idx = 0
        elif dr == forward_dr and dc == right_dc: dir_idx = 2
        else: return None
        piece_offset = 0 if move.promotion == chess.KNIGHT else (3 if move.promotion == chess.BISHOP else 6)
        return 64 + piece_offset + dir_idx, tr, tc
    return None

# ==========================================
# 3. CACHED ENGINE CONTROLLERS
# ==========================================

@st.cache_resource
def load_fischer_model():
    model = FischerNet()
    path = "chess_resnet.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_master_model():
    model = DualHeadChessNet()
    path = "master_resnet.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def execute_ai_prediction(engine_choice, f_model, m_model, board, temperature=0.1):
    """Unified inference router that maps correct outputs per engine model context."""
    tensor = board_to_tensor(board).unsqueeze(0)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    with torch.no_grad():
        if engine_choice == "Bobby Fischer AI":
            logits = f_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            eval_score = None  # Model 1 has no evaluation head
        else:
            logits, value_tensor = m_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            eval_score = float(value_tensor.item()) # Float between -1.0 and +1.0

    move_probs = []
    for move in legal_moves:
        idx = move_to_policy_index(move)
        move_probs.append(probs[idx[0]*64 + idx[1]*8 + idx[2]] if idx is not None else 0.0)
    
    move_probs = np.array(move_probs)
    if move_probs.sum() > 0:
        move_probs /= move_probs.sum()
    else:
        return random.choice(legal_moves), eval_score
        
    if temperature > 0:
        move_probs = np.exp(np.log(move_probs + 1e-9) / temperature)
        selected_move = np.random.choice(legal_moves, p=move_probs / move_probs.sum())
    else:
        selected_move = legal_moves[np.argmax(move_probs)]
        
    return selected_move, eval_score

# ==========================================
# 4. STREAMLIT FRAMEWORK GRAPHICS
# ==========================================

st.set_page_config(page_title="Grandmaster Arena AI", page_icon="👑", layout="centered")

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "player_color" not in st.session_state:
    st.session_state.player_color = "White"
if "selected_square" not in st.session_state:
    st.session_state.selected_square = None
if "current_eval" not in st.session_state:
    st.session_state.current_eval = 0.0

# Warm up model memories
fischer_engine = load_fischer_model()
master_engine = load_master_model()

st.title("👑 Grandmaster AI Arena")
st.write("Toggle between distinct neural networks to test your strategies.")

# Sidebar Navigation Panel
st.sidebar.title("Match Director")
engine_mode = st.sidebar.selectbox("Opponent Architecture:", ["Bobby Fischer AI", "World Elite Master Engine"])

# Status display of files inside the sidebar
if engine_mode == "Bobby Fischer AI":
    if os.path.exists("chess_resnet.pth"): st.sidebar.success("♟️ Fischer configuration active.")
    else: st.sidebar.warning("⚙️ Running Fischer via raw weights.")
else:
    if os.path.exists("master_resnet.pth"): st.sidebar.success("🧠 Master Dual-Head active.")
    else: st.sidebar.warning("⚙️ Running Master via raw weights.")

color_selection = st.sidebar.selectbox("Your Pieces:", ["White", "Black"])

if color_selection != st.session_state.player_color:
    st.session_state.player_color = color_selection
    st.session_state.board = chess.Board()
    st.session_state.selected_square = None
    st.rerun()

if st.sidebar.button("🔄 Reset Arena Board"):
    st.session_state.board = chess.Board()
    st.session_state.selected_square = None
    st.session_state.current_eval = 0.0
    st.rerun()

board = st.session_state.board
is_game_over = board.is_game_over(claim_draw=True)
is_ai_turn = (board.turn == chess.WHITE and st.session_state.player_color == "Black") or \
             (board.turn == chess.BLACK and st.session_state.player_color == "White")

# Render Master Board View
def render_svg_board(chess_board):
    flipped = (st.session_state.player_color == "Black")
    svg_data = chess.svg.board(board=chess_board, size=400, flipped=flipped)
    b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    st.markdown(
        f'<div style="display: flex; justify-content: center; margin-bottom: 15px;"><img src="data:image/svg+xml;base64,{b64_svg}" width="400"/></div>',
        unsafe_allow_html=True
    )

render_svg_board(board)

# Live Value Head Position Evaluator
if engine_mode == "World Elite Master Engine" and not is_game_over:
    ev = st.session_state.current_eval
    # Normalize score for clear human understanding
    perspective_eval = ev if st.session_state.player_color == "White" else -ev
    
    if perspective_eval > 0.15:
        st.metric(label="📊 Master AI Position Evaluation", value=f"+{perspective_eval:.2f}", delta="Advantage: User")
    elif perspective_eval < -0.15:
        st.metric(label="📊 Master AI Position Evaluation", value=f"{perspective_eval:.2f}", delta="-Advantage: AI", delta_color="inverse")
    else:
        st.metric(label="📊 Master AI Position Evaluation", value=f"{perspective_eval:.2f}", delta="Equal Position", delta_color="off")

st.write("---")

# Turn Sequence Core Loops
if not is_game_over:
    if is_ai_turn:
        st.info(f"🤖 **{engine_mode} is calculating alternatives...**")
        ai_move, calculated_eval = execute_ai_prediction(engine_mode, fischer_engine, master_engine, board, temperature=0.1)
        if ai_move:
            board.push(ai_move)
            if calculated_eval is not None:
                st.session_state.current_eval = calculated_eval
            st.rerun()
    else:
        if st.session_state.selected_square is not None:
            st.success(f"🟢 Selected **{chess.square_name(st.session_state.selected_square).upper()}**. Select target square.")
        else:
            st.success("🟢 **Your Move:** Click a piece from the grid matrix controller below.")

        # ==========================================
        # 5. CONTROL GRID CONTROLLER MATRIX
        # ==========================================
        files, ranks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['1', '2', '3', '4', '5', '6', '7', '8']
        unicode_pieces = {
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
        }

        rank_indices = range(7, -1, -1) if st.session_state.player_color == "White" else range(8)
        file_indices = range(8) if st.session_state.player_color == "White" else range(7, -1, -1)

        st.write("### 🎛️ Interactive Controller Matrix")
        for r in rank_indices:
            cols = st.columns(8)
            for i, c in enumerate(file_indices):
                sq_idx = r * 8 + c
                piece = board.piece_at(sq_idx)
                sq_name = files[c] + ranks[r]
                
                lbl = unicode_pieces[piece.symbol()] if piece else "·"
                button_text = f"**{lbl}**\n{sq_name.upper()}"
                
                if st.session_state.selected_square == sq_idx:
                    button_text = f"👉 {sq_name.upper()}"

                if cols[i].button(button_text, key=f"cell_{sq_idx}", use_container_width=True):
                    if st.session_state.selected_square is None:
                        player_is_white = (st.session_state.player_color == "White")
                        if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                            st.session_state.selected_square = sq_idx
                            st.rerun()
                    else:
                        from_sq = st.session_state.selected_square
                        to_sq = sq_idx
                        
                        move = chess.Move(from_sq, to_sq)
                        if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN and r in [0, 7]:
                            move.promotion = chess.QUEEN
                        
                        if move in board.legal_moves:
                            board.push(move)
                            st.session_state.selected_square = None
                            
                            # Run quick evaluation sync if switching turns to master context
                            if engine_mode == "World Elite Master Engine":
                                _, sync_eval = execute_ai_prediction(engine_mode, fischer_engine, master_engine, board, temperature=0.0)
                                if sync_eval is not None:
                                    st.session_state.current_eval = sync_eval
                            st.rerun()
                        else:
                            player_is_white = (st.session_state.player_color == "White")
                            if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                                st.session_state.selected_square = sq_idx
                            else:
                                st.session_state.selected_square = None
                            st.rerun()
else:
    st.error(f"🏁 **Game Over! Arena Result: {board.result()}**")
    st.session_state.selected_square = None

# ==========================================
# 6. CRASH-PROOF CHESS MOVE HISTORIAN
# ==========================================
if len(board.move_stack) > 0:
    st.write("---")
    with st.expander("📊 Complete Match History Log"):
        replay_board = chess.Board()
        safe_moves_san = []
        for historical_move in board.move_stack:
            safe_moves_san.append(replay_board.san(historical_move))
            replay_board.push(historical_move)
            
        formatted_history = []
        for i in range(0, len(safe_moves_san), 2):
            move_num = (i // 2) + 1
            w_move = safe_moves_san[i]
            b_move = safe_moves_san[i+1] if (i+1) < len(safe_moves_san) else ""
            formatted_history.append(f"**{move_num}.** {w_move} &nbsp;&nbsp;&nbsp;&nbsp; {b_move}")
        st.markdown("<br>".join(formatted_history), unsafe_allow_html=True)