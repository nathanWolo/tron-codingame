//! Bitboard Voronoi vs independent cell-BFS, plus 1v1 search smoke tests.
//! Compiled only by `cargo test`; not pasted into the CodinGame IDE.
use super::*;
use super::local::{bot_random, greedy_direction, random_start, XorShift};


    /// Per-player BFS distances from empty neighbours of `player`’s head.
    fn bfs_distances(state: &State, player: usize) -> [u16; BOARD_CELLS] {
    let mut distance = [UNREACHABLE; BOARD_CELLS];
    if !state.is_alive(player) {
        return distance;
    }
    let mut queue = [0u16; BOARD_CELLS];
    let mut queue_head = 0usize;
    let mut queue_tail = 0usize;
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    for dir in 0..4 {
        let col = head_col + DIR_X[dir];
        let row = head_row + DIR_Y[dir];
        if in_bounds(col, row) && !state.occupied.is_set(col, row) {
            let index = cell_index(col, row);
            distance[index] = 1;
            queue[queue_tail] = index as u16;
            queue_tail += 1;
        }
    }
    while queue_head < queue_tail {
        let index = queue[queue_head] as usize;
        queue_head += 1;
        let (col, row) = coords_from_index(index);
        let next_dist = distance[index] + 1;
        for dir in 0..4 {
            let next_col = col + DIR_X[dir];
            let next_row = row + DIR_Y[dir];
            if in_bounds(next_col, next_row) && !state.occupied.is_set(next_col, next_row) {
                let next_index = cell_index(next_col, next_row);
                if distance[next_index] == UNREACHABLE {
                    distance[next_index] = next_dist;
                    queue[queue_tail] = next_index as u16;
                    queue_tail += 1;
                }
            }
        }
    }
    distance
}

/// Original cell-by-cell Voronoi (independent BFS, then min-distance owner).
fn compute_voronoi_cell(state: &State) -> Voronoi {
    let player_count = state.player_count as usize;
    let mut distance = [[UNREACHABLE; BOARD_CELLS]; MAX_PLAYERS];
    for player in 0..player_count {
        if state.is_alive(player) {
            distance[player] = bfs_distances(state, player);
        }
    }
    let mut territory = [0i32; MAX_PLAYERS];
    let mut edge_sum = [0i32; MAX_PLAYERS];
    let mut reachable = [0i32; MAX_PLAYERS];
    let mut still_connected = false;
    let mut owned = [[0u32; 20]; MAX_PLAYERS];
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            if state.occupied.is_set(col, row) {
                continue;
            }
            let index = cell_index(col, row);
            let mut best_dist = UNREACHABLE;
            let mut best_player = -1i8;
            let mut tie_count = 0;
            let mut reacher_count = 0;
            for player in 0..player_count {
                if !state.is_alive(player) {
                    continue;
                }
                let dist = distance[player][index];
                if dist < UNREACHABLE {
                    reachable[player] += 1;
                    reacher_count += 1;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_player = player as i8;
                    tie_count = 1;
                } else if dist == best_dist && dist < UNREACHABLE {
                    tie_count += 1;
                }
            }
            if reacher_count >= 2 {
                still_connected = true;
            }
            if tie_count == 1 {
                let who = best_player as usize;
                territory[who] += 1;
                edge_sum[who] += empty_degree(state, col, row);
                owned[who][row as usize] |= 1u32 << col;
            }
        }
    }
    Voronoi {
        territory,
        edge_sum,
        reachable,
        still_connected,
        owned,
    }
}

fn shares_space_cell(state: &State, our_id: usize, opponent: usize) -> bool {
    if !state.is_alive(our_id) || !state.is_alive(opponent) {
        return false;
    }
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    let opp_col = state.head_x[opponent] as i32;
    let opp_row = state.head_y[opponent] as i32;
    if (our_col - opp_col).abs() + (our_row - opp_row).abs() == 1 {
        return true;
    }
    let dist = bfs_distances(state, our_id);
    for dir in 0..4 {
        let col = opp_col + DIR_X[dir];
        let row = opp_row + DIR_Y[dir];
        if in_bounds(col, row)
            && !state.occupied.is_set(col, row)
            && dist[cell_index(col, row)] < UNREACHABLE
        {
            return true;
        }
    }
    false
}

fn battlefront_cell(our_owned: &[u32; 20], opp_owned: &[u32; 20]) -> i32 {
    let mut front = 0;
    for row in 0..20 {
        for col in 0..30 {
            if our_owned[row] & (1u32 << col) == 0 {
                continue;
            }
            for dir in 0..4 {
                let next_col = col as i32 + DIR_X[dir];
                let next_row = row as i32 + DIR_Y[dir];
                if in_bounds(next_col, next_row)
                    && opp_owned[next_row as usize] & (1u32 << next_col) != 0
                {
                    front += 1;
                    break;
                }
            }
        }
    }
    front
}

fn owned_diff(fast: &Voronoi, slow: &Voronoi) -> String {
    let mut out = String::new();
    for player in 0..MAX_PLAYERS {
        for row in 0..20 {
            let xor = fast.owned[player][row] ^ slow.owned[player][row];
            if xor == 0 {
                continue;
            }
            for col in 0..30 {
                if xor & (1u32 << col) != 0 {
                    out.push_str(&format!(
                        "  p{player} ({col},{row}) fast={} slow={}\n",
                        (fast.owned[player][row] >> col) & 1,
                        (slow.owned[player][row] >> col) & 1
                    ));
                }
            }
        }
    }
    if out.is_empty() {
        out.push_str("  (owned bitboards match)\n");
    }
    out
}

fn main_opponent_cell(state: &State, our_id: usize) -> Option<usize> {
    let player_count = state.player_count as usize;
    let mut best_player = None;
    let mut best_dist = UNREACHABLE;
    let dist_map = bfs_distances(state, our_id);
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) {
            continue;
        }
        let head_col = state.head_x[player] as i32;
        let head_row = state.head_y[player] as i32;
        let mut dist = UNREACHABLE;
        for dir in 0..4 {
            let col = head_col + DIR_X[dir];
            let row = head_row + DIR_Y[dir];
            if !in_bounds(col, row) {
                continue;
            }
            if state.occupied.is_set(col, row)
                && col == state.head_x[our_id] as i32
                && row == state.head_y[our_id] as i32
            {
                dist = 0;
            } else if !state.occupied.is_set(col, row) {
                dist = dist.min(dist_map[cell_index(col, row)]);
            }
        }
        if dist < best_dist {
            best_dist = dist;
            best_player = Some(player);
        } else if dist == UNREACHABLE && best_dist == UNREACHABLE {
            let manhattan = (state.head_x[our_id] as i32 - head_col).abs()
                + (state.head_y[our_id] as i32 - head_row).abs();
            let previous = best_player.map(|previous_player| {
                (state.head_x[our_id] as i32 - state.head_x[previous_player] as i32).abs()
                    + (state.head_y[our_id] as i32 - state.head_y[previous_player] as i32).abs()
            });
            if best_player.is_none() || manhattan < previous.unwrap_or(i32::MAX) {
                best_player = Some(player);
            }
        }
    }
    best_player
}

fn eval_ffa_cell(state: &State, our_id: usize, ply: i32, slow: &Voronoi) -> i32 {
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if state.alive_mask.count_ones() == 1 {
        return MATE_SCORE - ply;
    }
    let player_count = state.player_count as usize;
    // Doom projection: rivals sealed into a pocket of at most FFA_DOOM_CELLS
    // (smaller than ours) are removed and their ribbons cleared, then the
    // cell-BFS Voronoi is recomputed on that board.
    let mut doomed = 0u8;
    for player in 0..player_count {
        if player != our_id
            && state.is_alive(player)
            && slow.reachable[player] <= FFA_DOOM_CELLS
            && slow.reachable[player] < slow.reachable[our_id]
        {
            doomed |= 1 << player;
        }
    }
    let projected;
    let slow = if doomed != 0 {
        let mut ghost = *state;
        for player in 0..player_count {
            if doomed & (1 << player) != 0 {
                ghost.kill(player);
            }
        }
        projected = compute_voronoi_cell(&ghost);
        &projected
    } else {
        slow
    };
    let mut best_other_territory = 0;
    let mut best_other_reach = 0;
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) || doomed & (1 << player) != 0 {
            continue;
        }
        best_other_territory = best_other_territory.max(slow.territory[player]);
        best_other_reach = best_other_reach.max(slow.reachable[player]);
    }
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    let center_penalty = -((our_col - 14).abs() + (our_row - 9).abs());
    let occupied_count = mask_popcount(&state.occupied.bits);
    let center_weight = (500 - occupied_count).max(0) / 80;
    slow.reachable[our_id] * 40 + slow.territory[our_id] * 25 - best_other_territory * 10
        + -best_other_reach * 4
        + slow.edge_sum[our_id] * 12
        + mobility(state, our_id) * 20
        + center_penalty * center_weight
}

fn dump_board(state: &State) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "players={} alive={:04b} heads",
        state.player_count, state.alive_mask
    ));
    for player in 0..state.player_count as usize {
        if state.is_alive(player) {
            out.push_str(&format!(
                " {}=({},{})",
                player, state.head_x[player], state.head_y[player]
            ));
        }
    }
    out.push('\n');
    for row in 0..20 {
        for col in 0..30 {
            let mut ch = if state.occupied.is_set(col, row) {
                '#'
            } else {
                '.'
            };
            for player in 0..state.player_count as usize {
                if state.is_alive(player)
                    && state.head_x[player] as i32 == col
                    && state.head_y[player] as i32 == row
                {
                    ch = char::from(b'A' + player as u8);
                }
            }
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

fn voronoi_gap(state: &State) -> Option<String> {
    let fast = compute_voronoi(state);
    let slow = compute_voronoi_cell(state);
    if fast.territory != slow.territory
        || fast.edge_sum != slow.edge_sum
        || fast.reachable != slow.reachable
        || fast.still_connected != slow.still_connected
        || fast.owned != slow.owned
    {
        return Some(format!(
            "voronoi mismatch\nfast terr={:?} edge={:?} reach={:?} conn={}\nslow terr={:?} edge={:?} reach={:?} conn={}\nowned diffs:\n{}{}",
            fast.territory,
            fast.edge_sum,
            fast.reachable,
            fast.still_connected,
            slow.territory,
            slow.edge_sum,
            slow.reachable,
            slow.still_connected,
            owned_diff(&fast, &slow),
            dump_board(state)
        ));
    }
    let player_count = state.player_count as usize;
    for our_id in 0..player_count {
        if !state.is_alive(our_id) {
            continue;
        }
        for opponent in 0..player_count {
            if opponent == our_id || !state.is_alive(opponent) {
                continue;
            }
            let fast_share = shares_space(state, our_id, opponent);
            let slow_share = shares_space_cell(state, our_id, opponent);
            if fast_share != slow_share {
                return Some(format!(
                    "shares_space {our_id} vs {opponent}: fast={fast_share} slow={slow_share}\n{}",
                    dump_board(state)
                ));
            }
            let fast_front = battlefront(&fast.owned[our_id], &fast.owned[opponent]);
            let slow_front = battlefront_cell(&slow.owned[our_id], &slow.owned[opponent]);
            if fast_front != slow_front {
                return Some(format!(
                    "battlefront {our_id} vs {opponent}: fast={fast_front} slow={slow_front}\n{}",
                    dump_board(state)
                ));
            }
        }
        let fast_opp = main_opponent(state, our_id);
        let slow_opp = main_opponent_cell(state, our_id);
        if fast_opp != slow_opp {
            return Some(format!(
                "main_opponent {our_id}: fast={fast_opp:?} slow={slow_opp:?}\n{}",
                dump_board(state)
            ));
        }
    }
    let empty = empty_rows(state);
    let mut scratch = Scratch::new();
    for player in 0..player_count {
        if !state.is_alive(player) {
            continue;
        }
        let flood = mask_popcount(&flood_mask(
            &head_seed(state, player, &empty),
            &empty,
        ));
        let cell_flood = flood_count(state, player, &mut scratch);
        if flood != cell_flood {
            return Some(format!(
                "flood {player}: mask={flood} bfs={cell_flood}\n{}",
                dump_board(state)
            ));
        }
        let fast_d = bitboard_distances(state, player);
        let slow_d = bfs_distances(state, player);
        if fast_d != slow_d {
            return Some(format!(
                "distances {player} bitboard != cell BFS\n{}",
                dump_board(state)
            ));
        }
    }
    if player_count >= 2 && state.is_alive(0) && state.is_alive(1) {
        let mut scratch = Scratch::new();
        let fast_eval = eval_1v1(state, 0, 1, 0, &mut scratch);
        let slow_share = shares_space_cell(state, 0, 1);
        let slow_eval = if !state.is_alive(0) {
            -MATE_SCORE
        } else if !state.is_alive(1) {
            MATE_SCORE
        } else if !slow_share {
            let our_fill = approx_fill(state, 0, &mut scratch);
            let opp_fill = approx_fill(state, 1, &mut scratch);
            let fill_diff = our_fill - opp_fill;
            let hug = wall_neighbor_count(
                state,
                state.head_x[0] as i32,
                state.head_y[0] as i32,
            ) - wall_neighbor_count(
                state,
                state.head_x[1] as i32,
                state.head_y[1] as i32,
            );
            fill_diff.signum() * 80
                + fill_diff * 60
                + hug * 5
                + (mobility(state, 0) - mobility(state, 1))
        } else {
            let territory = slow.territory[0] - slow.territory[1];
            let edges = slow.edge_sum[0] - slow.edge_sum[1];
            let our_mobility = mobility(state, 0);
            let opp_mobility = mobility(state, 1);
            let front = battlefront_cell(&slow.owned[0], &slow.owned[1]);
            let head_col = state.head_x[0] as i32;
            let head_row = state.head_y[0] as i32;
            let center_penalty = -((head_col - 14).abs() + (head_row - 9).abs());
            let occupied_count: i32 = (0..20)
                .map(|row| state.occupied.bits[row].count_ones())
                .sum::<u32>() as i32;
            let center_weight = (500 - occupied_count).max(0) / 80;
            territory * 50
                + edges * 12
                + (our_mobility - opp_mobility) * 6
                + front * 4
                + center_penalty * center_weight
        };
        if fast_eval != slow_eval {
            return Some(format!(
                "eval_1v1 fast={fast_eval} slow={slow_eval} share={slow_share}\n{}",
                dump_board(state)
            ));
        }
    }
    if player_count >= 3 {
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            let fast_eval = eval_ffa(state, player, 0);
            let slow_eval = eval_ffa_cell(state, player, 0, &slow);
            if fast_eval != slow_eval {
                return Some(format!(
                    "eval_ffa {player}: fast={fast_eval} slow={slow_eval}\n{}",
                    dump_board(state)
                ));
            }
        }
    }
    None
}

fn occupy_player(state: &mut State, player: usize, col: i32, row: i32) {
    state.occupy(player, col, row);
}

fn two_player_at(a: (i32, i32), b: (i32, i32)) -> State {
    let mut state = State::new(2);
    occupy_player(&mut state, 0, a.0, a.1);
    occupy_player(&mut state, 1, b.0, b.1);
    state
}

fn add_walls(state: &mut State, cells: &[(i32, i32)]) {
    for &(col, row) in cells {
        if !in_bounds(col, row) {
            continue;
        }
        let mut on_head = false;
        for player in 0..state.player_count as usize {
            if state.is_alive(player)
                && state.head_x[player] as i32 == col
                && state.head_y[player] as i32 == row
            {
                on_head = true;
            }
        }
        if !on_head {
            state.add_wall(col, row);
        }
    }
}

fn check_or_panic(state: &State, label: &str) {
    if let Some(msg) = voronoi_gap(state) {
        panic!("{label}: {msg}");
    }
}

fn assert_same_state(before: &State, after: &State, label: &str) {
    assert_eq!(before.occupied.bits, after.occupied.bits, "{label}: occupied");
    assert_eq!(before.head_x, after.head_x, "{label}: head_x");
    assert_eq!(before.head_y, after.head_y, "{label}: head_y");
    assert_eq!(before.alive_mask, after.alive_mask, "{label}: alive");
    for player in 0..MAX_PLAYERS {
        assert_eq!(
            before.trail[player].bits, after.trail[player].bits,
            "{label}: trail {player}"
        );
    }
    assert_eq!(before.hash, after.hash, "{label}: hash");
    assert_eq!(after.hash, after.full_hash(), "{label}: incremental hash");
}

fn bitboard_distances(state: &State, player: usize) -> [u16; BOARD_CELLS] {
    let mut distance = [UNREACHABLE; BOARD_CELLS];
    if !state.is_alive(player) {
        return distance;
    }
    let empty = empty_rows(state);
    let mut frontier = head_seed(state, player, &empty);
    let mut seen = frontier;
    let mut dist = 1u16;
    loop {
        let mut any = false;
        for row in 0..20 {
            let mut bits = frontier[row];
            while bits != 0 {
                let col = bits.trailing_zeros();
                bits &= bits - 1;
                distance[cell_index(col as i32, row as i32)] = dist;
                any = true;
            }
        }
        if !any {
            break;
        }
        let expanded = expand_mask(&frontier, &empty);
        let mut next = [0u32; 20];
        let mut grew = false;
        for row in 0..20 {
            next[row] = expanded[row] & !seen[row];
            if next[row] != 0 {
                grew = true;
            }
            seen[row] |= next[row];
        }
        if !grew {
            break;
        }
        frontier = next;
        dist += 1;
    }
    distance
}

fn oneply_best(state: &mut State, player: usize, opponent: usize, scratch: &mut Scratch) -> (u8, i32) {
    let (legal, move_count) = state.legal_moves(player);
    assert!(move_count > 0);
    let mut best_dir = legal[0];
    let mut best_score = i32::MIN;
    for i in 0..move_count {
        let old_col = state.head_x[player];
        let old_row = state.head_y[player];
        state.apply(player, legal[i] as usize);
        let score = eval_1v1(state, player, opponent, 1, scratch);
        state.undo_step(player, old_col, old_row);
        if score > best_score {
            best_score = score;
            best_dir = legal[i];
        }
    }
    (best_dir, best_score)
}

/// Root 1v1 search score from `our_id`’s seat (same window as [`choose_move`]).
fn root_negamax_score(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    depth: i32,
    scratch: &mut Scratch,
) -> i32 {
    let (moves, move_count) = state.legal_moves(our_id);
    assert!(move_count > 0, "root_negamax_score needs a legal move");
    let mut search = Search::new(std::time::Duration::from_secs(5));
    let mut best = -MATE_SCORE * 2;
    for i in 0..move_count {
        let old_col = state.head_x[our_id];
        let old_row = state.head_y[our_id];
        state.apply(our_id, moves[i] as usize);
        let score = -negamax_1v1(
            state,
            our_id,
            opponent,
            opponent,
            depth - 1,
            1,
            -MATE_SCORE * 2,
            -best,
            moves[i],
            NO_MOVE,
            QS_MAX,
            &mut search,
            scratch,
        );
        state.undo_step(our_id, old_col, old_row);
        assert!(!search.timed_out, "test search timed out");
        if score > best {
            best = score;
        }
    }
    best
}

fn walk_minimax_tree(state: &mut State, ply: i32, to_move: usize, label: &str) {
    check_or_panic(state, label);
    let player_count = state.player_count as usize;
    for player in 0..player_count {
        if !state.is_alive(player) {
            continue;
        }
        let fast_d = bitboard_distances(state, player);
        let slow_d = bfs_distances(state, player);
        if fast_d != slow_d {
            panic!("{label}: bitboard distances != BFS for player {player}\n{}", dump_board(state));
        }
    }
    if ply <= 0 {
        return;
    }
    if !state.is_alive(to_move) {
        return;
    }
    let snapshot = *state;
    let (moves, n) = state.legal_moves(to_move);
    if n == 0 {
        return;
    }
    let next = (to_move + 1) % player_count;
    for i in 0..n {
        let old_col = state.head_x[to_move];
        let old_row = state.head_y[to_move];
        state.apply(to_move, moves[i] as usize);
        walk_minimax_tree(state, ply - 1, next, label);
        state.undo_step(to_move, old_col, old_row);
        assert_same_state(&snapshot, state, label);
    }
}

fn random_play_state(rng: &mut XorShift, player_count: usize, plies: u32) -> State {
    let mut state = random_start(rng, player_count);
    for _ in 0..plies {
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            if let Some(dir) = bot_random(&state, player, rng) {
                state.apply(player, dir as usize);
            } else {
                state.kill(player);
            }
        }
    }
    assert_eq!(state.hash, state.full_hash(), "random play: incremental hash");
    state
}

fn random_wall_state(rng: &mut XorShift, player_count: usize, walls: u32) -> State {
    let mut state = random_start(rng, player_count);
    let mut added = 0u32;
    let mut tries = 0u32;
    while added < walls && tries < walls * 8 {
        tries += 1;
        let col = rng.gen_range(WIDTH as u32) as i32;
        let row = rng.gen_range(HEIGHT as u32) as i32;
        if state.occupied.is_set(col, row) {
            continue;
        }
        state.add_wall(col, row);
        added += 1;
    }
    state
}

    #[test]
    fn statement_spawn() {
        check_or_panic(&two_player_at((9, 5), (10, 7)), "statement");
    }

    #[test]
    fn adjacent_heads() {
        check_or_panic(&two_player_at((14, 9), (15, 9)), "adjacent");
        check_or_panic(&two_player_at((0, 0), (1, 0)), "adjacent-corner");
    }

    #[test]
    fn corners_and_edges() {
        check_or_panic(&two_player_at((0, 0), (29, 19)), "corners");
        check_or_panic(&two_player_at((0, 9), (29, 9)), "row-ends");
        check_or_panic(&two_player_at((14, 0), (14, 19)), "col-ends");
    }

    #[test]
    fn corridor_and_split() {
        let mut corridor = two_player_at((0, 10), (29, 10));
        for col in 1..29 {
            add_walls(&mut corridor, &[(col, 9), (col, 11)]);
        }
        check_or_panic(&corridor, "corridor");
        let mut split = two_player_at((5, 10), (25, 10));
        for row in 0..20 {
            add_walls(&mut split, &[(15, row)]);
        }
        check_or_panic(&split, "split");
    }

    #[test]
    fn boxed_player() {
        let mut state = two_player_at((5, 5), (20, 10));
        add_walls(&mut state, &[(5, 4), (5, 6), (4, 5), (6, 5)]);
        check_or_panic(&state, "boxed");
    }

    #[test]
    fn three_and_four_players() {
        let mut three = State::new(3);
        occupy_player(&mut three, 0, 2, 2);
        occupy_player(&mut three, 1, 27, 2);
        occupy_player(&mut three, 2, 14, 17);
        check_or_panic(&three, "3p");
        let mut four = State::new(4);
        occupy_player(&mut four, 0, 1, 1);
        occupy_player(&mut four, 1, 28, 1);
        occupy_player(&mut four, 2, 1, 18);
        occupy_player(&mut four, 3, 28, 18);
        check_or_panic(&four, "4p");
    }

    #[test]
    fn random_play_and_children() {
        let mut rng = XorShift::new(1);
        for player_count in [2usize, 3, 4] {
            for _game in 0..200 {
                let plies = rng.gen_range(50);
                let state = random_play_state(&mut rng, player_count, plies);
                check_or_panic(&state, "random-play");
                for player in 0..player_count {
                    if !state.is_alive(player) {
                        continue;
                    }
                    let (moves, n) = state.legal_moves(player);
                    for i in 0..n {
                        let mut child = state;
                        child.apply(player, moves[i] as usize);
                        check_or_panic(&child, "random-child");
                    }
                }
            }
        }
    }

    #[test]
    fn random_walls() {
        let mut rng = XorShift::new(2);
        for player_count in [2usize, 3, 4] {
            for _ in 0..250 {
                let walls = 20 + rng.gen_range(120);
                check_or_panic(
                    &random_wall_state(&mut rng, player_count, walls),
                    "random-walls",
                );
            }
        }
    }

    #[test]
    fn no_horizontal_or_vertical_wrap() {
        let mut horizontal = two_player_at((0, 0), (29, 0));
        for row in 0..20 {
            for col in 1..29 {
                add_walls(&mut horizontal, &[(col, row)]);
            }
        }
        assert!(
            !shares_space(&horizontal, 0, 1),
            "col 0 must not wrap to col 29\n{}",
            dump_board(&horizontal)
        );
        check_or_panic(&horizontal, "no-hwrap");

        let mut vertical = two_player_at((5, 0), (5, 19));
        for row in 1..19 {
            for col in 0..30 {
                add_walls(&mut vertical, &[(col, row)]);
            }
        }
        assert!(
            !shares_space(&vertical, 0, 1),
            "row 0 must not wrap to row 19\n{}",
            dump_board(&vertical)
        );
        check_or_panic(&vertical, "no-vwrap");
    }

    #[test]
    fn checkerboard_and_sparse_walls() {
        let mut checker = two_player_at((1, 0), (28, 19));
        for row in 0..20 {
            for col in 0..30 {
                if (col + row) % 2 == 0 {
                    add_walls(&mut checker, &[(col, row)]);
                }
            }
        }
        check_or_panic(&checker, "checkerboard");

        let mut sparse = two_player_at((0, 0), (29, 19));
        for row in 0..20 {
            for col in 0..30 {
                if col % 3 == 1 && row % 2 == 0 {
                    add_walls(&mut sparse, &[(col, row)]);
                }
            }
        }
        check_or_panic(&sparse, "sparse-lattice");
    }

    #[test]
    fn maze_rooms_and_choke() {
        let mut rooms = two_player_at((2, 2), (27, 17));
        for row in 0..20 {
            add_walls(&mut rooms, &[(14, row), (15, row)]);
        }
        add_walls(&mut rooms, &[(14, 10)]);
        rooms.occupied.clear(14, 10);
        rooms.occupied.clear(15, 10);
        check_or_panic(&rooms, "two-rooms-choke");

        let mut pockets = two_player_at((1, 1), (10, 10));
        for &(col, row) in &[
            (3, 0),
            (3, 1),
            (3, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
        ] {
            add_walls(&mut pockets, &[(col, row)]);
        }
        check_or_panic(&pockets, "boxed-corner-pocket");
    }

    #[test]
    fn snake_trails() {
        let mut state = two_player_at((0, 0), (29, 19));
        for row in 0..8 {
            for i in 0..30 {
                let col = if row % 2 == 0 { i } else { 29 - i };
                if (col, row) == (0, 0) || state.occupied.is_set(col, row) {
                    continue;
                }
                state.add_wall(col, row);
                state.trail[0].set(col, row);
                state.head_x[0] = col as i8;
                state.head_y[0] = row as i8;
            }
        }
        for row in (12..20).rev() {
            for i in 0..30 {
                let col = if row % 2 == 0 { i } else { 29 - i };
                if (col, row) == (29, 19) || state.occupied.is_set(col, row) {
                    continue;
                }
                state.add_wall(col, row);
                state.trail[1].set(col, row);
                state.head_x[1] = col as i8;
                state.head_y[1] = row as i8;
            }
        }
        check_or_panic(&state, "snake-trails");
    }

    #[test]
    fn greedy_full_games() {
        let mut rng = XorShift::new(0xABCDEF);
        let mut scratch = Scratch::new();
        for game in 0..40 {
            let mut state = random_start(&mut rng, 2);
            check_or_panic(&state, "greedy-start");
            for _turn in 0..500 {
                if state.alive_mask.count_ones() <= 1 {
                    break;
                }
                for player in 0..2 {
                    if !state.is_alive(player) {
                        continue;
                    }
                    check_or_panic(&state, "greedy-mid");
                    match greedy_direction(&state, player, &mut scratch) {
                        Some(dir) if state.is_legal_dir(player, dir as usize) => {
                            state.apply(player, dir as usize);
                        }
                        _ => state.kill(player),
                    }
                }
            }
            check_or_panic(&state, &format!("greedy-end-{game}"));
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn exhaustive_empty_board_pairs() {
        for a in 0..BOARD_CELLS {
            let (col_a, row_a) = coords_from_index(a);
            for b in 0..BOARD_CELLS {
                if a == b {
                    continue;
                }
                let (col_b, row_b) = coords_from_index(b);
                check_or_panic(
                    &two_player_at((col_a, row_a), (col_b, row_b)),
                    "empty-pair",
                );
            }
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn exhaustive_pairs_with_mid_wall() {
        for a in 0..BOARD_CELLS {
            let (col_a, row_a) = coords_from_index(a);
            if col_a == 15 {
                continue;
            }
            for b in 0..BOARD_CELLS {
                if a == b {
                    continue;
                }
                let (col_b, row_b) = coords_from_index(b);
                if col_b == 15 {
                    continue;
                }
                let mut state = two_player_at((col_a, row_a), (col_b, row_b));
                for row in 0..20 {
                    add_walls(&mut state, &[(15, row)]);
                }
                check_or_panic(&state, "mid-wall-pair");
            }
        }
    }

    #[test]
    fn ffa_random_triples_and_quads() {
        let mut rng = XorShift::new(99);
        for _ in 0..800 {
            let mut three = State::new(3);
            let mut used = [false; BOARD_CELLS];
            for player in 0..3 {
                loop {
                    let index = rng.gen_range(BOARD_CELLS as u32) as usize;
                    if used[index] {
                        continue;
                    }
                    used[index] = true;
                    let (col, row) = coords_from_index(index);
                    occupy_player(&mut three, player, col, row);
                    break;
                }
            }
            check_or_panic(&three, "random-triple");
        }
        for _ in 0..400 {
            let mut four = State::new(4);
            let mut used = [false; BOARD_CELLS];
            for player in 0..4 {
                loop {
                    let index = rng.gen_range(BOARD_CELLS as u32) as usize;
                    if used[index] {
                        continue;
                    }
                    used[index] = true;
                    let (col, row) = coords_from_index(index);
                    occupy_player(&mut four, player, col, row);
                    break;
                }
            }
            check_or_panic(&four, "random-quad");
        }
    }

    #[test]
    fn kill_clears_and_reconnects() {
        let mut three = State::new(3);
        occupy_player(&mut three, 0, 5, 10);
        occupy_player(&mut three, 2, 25, 10);
        occupy_player(&mut three, 1, 15, 0);
        for row in 1..20 {
            three.add_wall(15, row);
            three.trail[1].set(15, row);
            three.head_x[1] = 15;
            three.head_y[1] = row as i8;
        }
        check_or_panic(&three, "three-split");
        assert!(
            !shares_space(&three, 0, 2),
            "player 1's trail should wall off the board\n{}",
            dump_board(&three)
        );
        three.kill(1);
        check_or_panic(&three, "after-kill-middle");
        assert!(
            shares_space(&three, 0, 2),
            "killing the middle bike should open their trail\n{}",
            dump_board(&three)
        );
    }

    #[test]
    fn minimax_tree_matches_and_undoes() {
        let mut starts = vec![
            two_player_at((9, 5), (10, 7)),
            two_player_at((0, 0), (29, 19)),
            two_player_at((14, 9), (15, 9)),
            two_player_at((5, 5), (24, 14)),
        ];
        let mut rng = XorShift::new(7);
        for _ in 0..8 {
            let plies = 8 + rng.gen_range(16);
            starts.push(random_play_state(&mut rng, 2, plies));
        }
        for (i, start) in starts.iter().enumerate() {
            let mut state = *start;
            walk_minimax_tree(&mut state, 4, 0, &format!("tree-{i}"));
            assert_same_state(start, &state, &format!("tree-root-{i}"));
        }
    }

    #[test]
    fn choose_move_restores_board() {
        let mut rng = XorShift::new(11);
        let mut scratch = Scratch::new();
        for i in 0..20 {
            let plies = rng.gen_range(24);
            let start = random_play_state(&mut rng, 2, plies);
            if start.alive_mask.count_ones() < 2 {
                continue;
            }
            if start.legal_moves(0).1 < 2 {
                continue;
            }
            let mut state = start;
            let _dir = choose_move(&mut state, 0, NO_MOVE, 15, &mut scratch);
            assert_same_state(&start, &state, &format!("choose-{i}"));
        }
        for i in 0..10 {
            let plies = rng.gen_range(16);
            let start = random_play_state(&mut rng, 4, plies);
            if start.alive_mask.count_ones() < 3 {
                continue;
            }
            if start.legal_moves(0).1 < 2 {
                continue;
            }
            let mut state = start;
            let _dir = choose_move(&mut state, 0, NO_MOVE, 15, &mut scratch);
            assert_same_state(&start, &state, &format!("choose-4p-{i}"));
        }
    }

    #[test]
    fn oneply_agrees_with_eval() {
        let mut rng = XorShift::new(13);
        let mut scratch = Scratch::new();
        for _ in 0..40 {
            let plies = rng.gen_range(20);
            let mut state = random_play_state(&mut rng, 2, plies);
            if !state.is_alive(0) || !state.is_alive(1) {
                continue;
            }
            if state.legal_moves(0).1 == 0 {
                continue;
            }
            check_or_panic(&state, "oneply-root");
            let before = state;
            let (_dir, _score) = oneply_best(&mut state, 0, 1, &mut scratch);
            assert_same_state(&before, &state, "oneply-undo");
        }
    }

    #[test]
    fn trapping_the_opponent_is_a_win() {
        // P1 is boxed in the corner. Any legal P0 step leaves them with no
        // reply, so depth-1 search must be a win — not `-MATE` after negate.
        let mut state = two_player_at((10, 10), (0, 0));
        add_walls(&mut state, &[(1, 0), (0, 1)]);
        assert_eq!(state.legal_moves(1).1, 0);
        assert!(state.legal_moves(0).1 > 0);
        let mut scratch = Scratch::new();
        let before = state;
        let score = root_negamax_score(&mut state, 0, 1, 1, &mut scratch);
        assert_same_state(&before, &state, "trap-undo");
        assert!(
            score >= MATE_SCORE - 200,
            "trapping the opponent scored {score}, expected a win"
        );
    }

    #[test]
    fn opening_search_is_not_mate() {
        let mut state = two_player_at((9, 5), (10, 7));
        let mut scratch = Scratch::new();
        let before = state;
        let shallow = root_negamax_score(&mut state, 0, 1, 1, &mut scratch);
        let deeper = root_negamax_score(&mut state, 0, 1, 8, &mut scratch);
        assert_same_state(&before, &state, "opening-undo");
        assert!(
            shallow.abs() < MATE_SCORE - 200,
            "depth-1 opening scored {shallow}"
        );
        assert!(
            deeper.abs() < MATE_SCORE - 200,
            "depth-8 opening scored {deeper}"
        );
    }

    #[test]
    fn quiescence_sees_forced_fill_death() {
        // P1 has one empty cell, then nowhere. Depth-1 static eval still sees
        // them alive; QS (mobility ≤ 1) should return a win for P0.
        let mut state = two_player_at((10, 10), (0, 0));
        add_walls(&mut state, &[(1, 0), (0, 2), (1, 1)]);
        assert_eq!(state.legal_moves(1).1, 1);
        let mut scratch = Scratch::new();
        let before = state;
        let score = root_negamax_score(&mut state, 0, 1, 1, &mut scratch);
        assert_same_state(&before, &state, "qs-undo");
        assert!(
            score >= MATE_SCORE - 200,
            "quiescence should see P1 die after their last cell, got {score}"
        );
    }

    /// Occupy every cell except the two heads and `keep`.
    fn wall_all_except(state: &mut State, keep: &[(i32, i32)]) {
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let mut skip = false;
                for player in 0..state.player_count as usize {
                    if state.is_alive(player)
                        && state.head_x[player] as i32 == col
                        && state.head_y[player] as i32 == row
                    {
                        skip = true;
                    }
                }
                if skip || keep.iter().any(|&cell| cell == (col, row)) {
                    continue;
                }
                state.add_wall(col, row);
            }
        }
    }

    #[test]
    fn checkerboard_path_bound_formula() {
        assert_eq!(checkerboard_path_bound(0, 0, 0), 0);
        assert_eq!(checkerboard_path_bound(0, 1, 0), 1);
        assert_eq!(checkerboard_path_bound(1, 0, 1), 1);
        assert_eq!(checkerboard_path_bound(0, 5, 5), 10);
        assert_eq!(checkerboard_path_bound(0, 6, 5), 11);
        assert_eq!(checkerboard_path_bound(0, 5, 6), 10);
        assert_eq!(checkerboard_path_bound(1, 5, 6), 11);
        assert_eq!(checkerboard_path_bound(1, 1, 4), 3);
    }

    #[test]
    fn rival_distances_match_cell_bfs() {
        let mut rng = XorShift::new(23);
        for i in 0..40 {
            let plies = rng.gen_range(30);
            let state = random_play_state(&mut rng, 4, plies);
            for our_id in 0..4 {
                if !state.is_alive(our_id) {
                    continue;
                }
                let fast = rival_distances(&state, our_id);
                let d = bfs_distances(&state, our_id);
                for rival in 0..4 {
                    if rival == our_id || !state.is_alive(rival) {
                        continue;
                    }
                    let (rc, rr) = (state.head_x[rival] as i32, state.head_y[rival] as i32);
                    let mut best = UNREACHABLE;
                    if heads_manhattan(&state, our_id, rival) == 1 {
                        best = 1;
                    } else {
                        for dir in 0..4 {
                            let (c, r) = (rc + DIR_X[dir], rr + DIR_Y[dir]);
                            if in_bounds(c, r) && !state.occupied.is_set(c, r) {
                                let dd = d[cell_index(c, r)];
                                if dd < UNREACHABLE {
                                    best = best.min(dd + 1);
                                }
                            }
                        }
                    }
                    assert_eq!(fast[rival], best, "state {i} our {our_id} rival {rival}");
                }
            }
        }
    }

    #[test]
    fn approx_fill_caps_3x3_minority_entry() {
        // 3×3 empty room: 5 even, 4 odd. Enter from (2,0) onto (2,1) (odd),
        // so a path can take at most 8 cells. Uncut DFS would count all 9.
        let mut state = two_player_at((2, 0), (29, 19));
        let mut room = Vec::new();
        for row in 1..4 {
            for col in 1..4 {
                room.push((col, row));
            }
        }
        wall_all_except(&mut state, &room);
        let mut scratch = Scratch::new();
        assert_eq!(flood_count(&state, 0, &mut scratch), 9);
        assert_eq!(checkerboard_reach_bound(&state, 0), 8);
        assert_eq!(approx_fill(&state, 0, &mut scratch), 8);
    }

    #[test]
    fn approx_fill_caps_plus_shape() {
        // Plus of 5 cells. The 8-ring local-cut test treats the centre as
        // connected (open_count == 4), so DFS would sum; checkerboard caps
        // a path from the stem at 3.
        let mut state = two_player_at((5, 3), (29, 19));
        wall_all_except(
            &mut state,
            &[(5, 4), (5, 5), (5, 6), (4, 5), (6, 5)],
        );
        let mut scratch = Scratch::new();
        assert_eq!(flood_count(&state, 0, &mut scratch), 5);
        assert_eq!(checkerboard_reach_bound(&state, 0), 3);
        assert_eq!(approx_fill(&state, 0, &mut scratch), 3);
    }

    #[test]
    fn checkerboard_bound_never_exceeds_flood() {
        let mut rng = XorShift::new(23);
        let mut scratch = Scratch::new();
        for _ in 0..40 {
            let plies = rng.gen_range(24);
            let state = random_play_state(&mut rng, 2, plies);
            for player in 0..2 {
                if !state.is_alive(player) {
                    continue;
                }
                let flood = flood_count(&state, player, &mut scratch);
                let bound = checkerboard_reach_bound(&state, player);
                let fill = approx_fill(&state, player, &mut scratch);
                assert!(
                    bound <= flood,
                    "player {player} bound {bound} > flood {flood}\n{}",
                    dump_board(&state)
                );
                assert!(
                    fill <= bound,
                    "player {player} fill {fill} > bound {bound}\n{}",
                    dump_board(&state)
                );
            }
        }
    }
