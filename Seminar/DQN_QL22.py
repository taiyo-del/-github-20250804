#海野さんプロトタイプ 3つの交差点　単一エージェント　海野さんの状態、行動、報酬　net,sumocfgのみ変更 マルチエージェント  結果の可視化 報酬変更追加 報酬のクリッピング 損失の可視化 DoubleDQN(update関数)変更  状態の追加(論文) 東西、南北を優先の二つの行動 優先車線切り替え10秒固定 リプレイバッファ共有　論文報酬＋デッドロック 黄色赤信号 ネットワークの拡張
#交差点ごとに独立した報酬　
#エージェントごとに独立したリプレイバッファ
#デッドロック　全体で1つの判定　どの交差点でデッドロックが起こったかは判別できない
#毎ステップで優先方向を切り替えられる
#交差点ごとに「別々の状態」を扱う 全交差点の状態をまとめて「1つの大きな状態」にはしていない
#各エージェント（J1, J4, J7）は自分の経験を SharedReplayBuffer(共有,共有のポインタ、グローバル変数) に入れる。・代表（central_trainerが共有バッファからまとめてサンプリングして学習する。・学習が終わったらその重みを全エージェントに同期する。
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
import math
import numpy as np
from collections import deque, defaultdict
import xml.etree.ElementTree as ET
from itertools import permutations
from ast import literal_eval
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci
import traci.constants as tc
import sumolib
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

DEPART = ["E0", "-E1", "-E2", "-E4","-E5", "-E8", "-E7", "-E9"]
ARRIVAL_E0 = ["E1", "E2", "E4","E5", "E8", "E7", "E9"]
ARRIVAL_E1 = ["-E0",  "E2", "E4","E5", "E8", "E7", "E9"]
ARRIVAL_E2 = ["-E0", "E1", "E4","E5", "E8", "E7", "E9"]
ARRIVAL_E4 = ["-E0", "E1", "E2", "E5", "E8", "E7", "E9"]
ARRIVAL_E5 = ["-E0", "E1", "E2", "E4", "E8", "E7", "E9"]
ARRIVAL_E8 = ["-E0", "E1", "E2", "E4","E5", "E7", "E9"]
ARRIVAL_E7 = ["-E0", "E1", "E2", "E4","E5", "E8",  "E9"]
ARRIVAL_E9 = ["-E0", "E1", "E2", "E4","E5", "E8", "E7"]

SPEED = 5
DISTANCE = SPEED * 10
net = sumolib.net.readNet('data/previous_reserch.net.xml')
JUNCTION_NODE1 = "J1"
JUNCTION_NODE2 = "J4"
JUNCTION_NODE3 = "J7"
JUNCTION_NODES = [JUNCTION_NODE1, JUNCTION_NODE2, JUNCTION_NODE3]
PRIORITY = {JUNCTION_NODE1: "E6", JUNCTION_NODE2: "-E2", JUNCTION_NODE3: "-E9"}

EW_EDGES = {
    JUNCTION_NODE1: ["-E3", "E0"],
    JUNCTION_NODE2: ["-E6", "E3"],
    JUNCTION_NODE3: ["-E9", "E6"],
}
NS_EDGES = {
    JUNCTION_NODE1: ["-E2", "-E1"],
    JUNCTION_NODE2: ["-E5", "-E4"],
    JUNCTION_NODE3: ["-E8", "-E7"],
}

NEIGHBOR_MAP = {
    JUNCTION_NODE1: {JUNCTION_NODE2: JUNCTION_NODE2},
    JUNCTION_NODE2: {JUNCTION_NODE1: JUNCTION_NODE1, JUNCTION_NODE3: JUNCTION_NODE3},
    JUNCTION_NODE3: {JUNCTION_NODE2: JUNCTION_NODE2}
}

STATE_FEATURE_CONFIG = {
    "local_use_current_phase": True,
    "local_use_vehicle_counts": True,
    "local_use_queue_length": True,
    "local_use_green_elapse": True,
    "neighbor_use_current_phase": False,
    "neighbor_use_vehicle_counts": False,
    "neighbor_use_queue_length": False,
    "neighbor_use_green_elapse": False,
    "use_neighboring_info": False,
    "detection_range": 200,
    "green_elapse_threshold": 36,
    "max_vehicles_normalize": 100,
    "max_queue_normalize": 200
}

current_virtual_phases = defaultdict(int)
virtual_phase_start_times = defaultdict(float)

class SharedReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, node_id, state, action, reward, next_state, done):
        # state と next_state は numpy array を想定
        self.buffer.append((node_id, state, action, reward, next_state, done))

    def sample(self):
        data = random.sample(self.buffer, self.batch_size)
        node_ids = np.array([x[0] for x in data])
        states = np.stack([x[1] for x in data]).astype(np.float32)
        actions = np.array([x[2] for x in data]).astype(np.long)
        rewards = np.array([x[3] for x in data]).astype(np.float32)
        next_states = np.stack([x[4] for x in data]).astype(np.float32)
        dones = np.array([x[5] for x in data]).astype(np.float32)
        return node_ids, states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------- Qネットワーク ----------
class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

# ---------- DQNAgent（shared replay を受け取る, 代表が学習） ----------
class DQNAgent:
    def __init__(self, state_size, action_size=2, shared_replay=True):
        self.gamma = 0.98
        self.lr = 0.0001
        self.epsilon = 0.2
        self.buffer_size = 10000
        self.batch_size = 64
        self.state_size = state_size
        self.action_size = action_size

        # 各エージェントは shared_replay を参照（None でも動くが学習は central_trainer に任せる）
        self.replay_buffer = shared_replay if shared_replay is not None else SharedReplayBuffer(self.buffer_size, self.batch_size)

        # ネットワーク（各エージェントは同アーキテクチャを持つがパラメータ同期で共有する方針）
        self.qnet = QNet(self.state_size, self.action_size)
        self.qnet_target = QNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_history = []

    def get_action(self, state, epsilon=None):#epsilon=数値
        # state: numpy array
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            s = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
            with torch.no_grad():
                qs = self.qnet(s)
            return int(qs.argmax().item())

    # central trainer が呼ぶ学習関数
    def update_from_shared(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        node_ids, states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a)
        q_vals = self.qnet(states_t)
        q = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_actions = self.qnet(next_states_t).argmax(1)
        next_qs_target = self.qnet_target(next_states_t)
        next_q = next_qs_target.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()

        target = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet_target.load_state_dict(self.qnet.state_dict())

def init(is_gui):
    if is_gui:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
    else:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
    sumoCmd = [sumoBinary, "-c", "data/3intersections_deadlock.sumocfg",]
    traci.start(sumoCmd)

def make_vehicle(vehID, routeID, depart_time):
    traci.vehicle.addLegacy(vehID, routeID, depart=depart_time)
    traci.vehicle.setSpeed(vehID, SPEED)
    traci.vehicle.setMaxSpeed(vehID, SPEED)

def make_random_route(num):
    ok = True
    while ok:
        depart = random.choice(DEPART)
        if depart=="E0":
            arrive = random.choice(ARRIVAL_E0)
        elif depart=="-E1":
            arrive = random.choice(ARRIVAL_E1)
        elif depart=="-E2":
            arrive = random.choice(ARRIVAL_E2)
        elif depart=="-E4":
            arrive = random.choice(ARRIVAL_E4)
        elif depart=="-E5":
            arrive = random.choice(ARRIVAL_E5)
        elif depart=="-E8":
            arrive = random.choice(ARRIVAL_E8)
        elif depart=="-E7":
            arrive = random.choice(ARRIVAL_E7)
        elif depart=="-E9":
            arrive = random.choice(ARRIVAL_E9)
        try:
            traci.route.add(f"random_route_{num}", [depart, arrive])
            ok = False
        except:
            pass
    return f"random_route_{num}"

def normalize_minmax(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def get_current_phase_feature(junction_id):
    virtual_phase = current_virtual_phases.get(junction_id, 0)
    return [0, 1] if virtual_phase == 0 else [1, 0]

def get_green_elapse_feature(junction_id):
    current_time = traci.simulation.getTime()
    start_time = virtual_phase_start_times.get(junction_id, current_time)
    elapsed = current_time - start_time
    return [1] if elapsed > STATE_FEATURE_CONFIG["green_elapse_threshold"] else [0]

def get_vehicle_counts_feature(node_id):
    vehicle_counts = []
    try:
        incoming_edges = net.getNode(node_id).getIncoming()
    except KeyError: return [0.0] * 4
    if not incoming_edges: return [0.0] * 4
    for edge in incoming_edges:
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge.getID())
            count = 0
            for vid in vehicles_on_edge:
                try:
                    veh_pos = traci.vehicle.getLanePosition(vid)
                    lane_id = traci.vehicle.getLaneID(vid)
                    if not lane_id or lane_id not in traci.lane.getIDList(): continue
                    edge_len = traci.lane.getLength(lane_id)
                    if edge_len - veh_pos <= STATE_FEATURE_CONFIG["detection_range"]:
                        count += 1
                except traci.TraCIException: pass
            norm_count = min(count / STATE_FEATURE_CONFIG["max_vehicles_normalize"], 1.0)
            vehicle_counts.append(norm_count)
        except traci.TraCIException: vehicle_counts.append(0.0)
    while len(vehicle_counts) < 4: vehicle_counts.append(0.0)
    return vehicle_counts[:4]

def get_queue_length_feature(node_id):
    queue_lengths = []
    try:
        incoming_edges = net.getNode(node_id).getIncoming()
    except KeyError: return [0.0] * 4
    if not incoming_edges: return [0.0] * 4
    for edge in incoming_edges:
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge.getID())
            qlen = 0
            for vid in vehicles_on_edge:
                try:
                    if traci.vehicle.getSpeed(vid) < 0.5:
                        veh_pos = traci.vehicle.getLanePosition(vid)
                        lane_id = traci.vehicle.getLaneID(vid)
                        if not lane_id or lane_id not in traci.lane.getIDList(): continue
                        edge_len = traci.lane.getLength(lane_id)
                        if edge_len - veh_pos <= STATE_FEATURE_CONFIG["detection_range"]:
                            qlen = max(qlen, edge_len - veh_pos)
                except traci.TraCIException: pass
            normalized = min(qlen / STATE_FEATURE_CONFIG["max_queue_normalize"], 1.0)
            queue_lengths.append(normalized)
        except traci.TraCIException: queue_lengths.append(0.0)
    while len(queue_lengths) < 4: queue_lengths.append(0.0)
    return queue_lengths[:4]

def get_paper_based_state(node_id, junction_id, neighboring_nodes=None):
    if neighboring_nodes is None:
        neighboring_nodes = NEIGHBOR_MAP.get(junction_id, [])
    state_vector = []
    if STATE_FEATURE_CONFIG["local_use_current_phase"]:
        state_vector.extend(get_current_phase_feature(junction_id))
    if STATE_FEATURE_CONFIG["local_use_vehicle_counts"]:
        state_vector.extend(get_vehicle_counts_feature(node_id))
    if STATE_FEATURE_CONFIG["local_use_queue_length"]:
        state_vector.extend(get_queue_length_feature(node_id))
    if STATE_FEATURE_CONFIG["local_use_green_elapse"]:
        state_vector.extend(get_green_elapse_feature(junction_id))

    num_single_neighbor_features = 0
    if STATE_FEATURE_CONFIG["neighbor_use_current_phase"]: num_single_neighbor_features += 2
    if STATE_FEATURE_CONFIG["neighbor_use_vehicle_counts"]: num_single_neighbor_features += 4
    if STATE_FEATURE_CONFIG["neighbor_use_queue_length"]: num_single_neighbor_features += 4
    if STATE_FEATURE_CONFIG["neighbor_use_green_elapse"]: num_single_neighbor_features += 1

    processed = 0
    if STATE_FEATURE_CONFIG["use_neighboring_info"] and neighboring_nodes:
        for neigh_junc, neigh_node in neighboring_nodes.items():
            if processed >= 2: break
            state_vector.append(1.0)
            if STATE_FEATURE_CONFIG["neighbor_use_current_phase"]:
                state_vector.extend(get_current_phase_feature(neigh_junc))
            if STATE_FEATURE_CONFIG["neighbor_use_vehicle_counts"]:
                state_vector.extend(get_vehicle_counts_feature(neigh_node))
            if STATE_FEATURE_CONFIG["neighbor_use_queue_length"]:
                state_vector.extend(get_queue_length_feature(neigh_node))
            if STATE_FEATURE_CONFIG["neighbor_use_green_elapse"]:
                state_vector.extend(get_green_elapse_feature(neigh_junc))
            processed += 1
    while processed < 2 and STATE_FEATURE_CONFIG["use_neighboring_info"]:
        state_vector.append(0.0)
        state_vector.extend([0.0] * num_single_neighbor_features)
        processed += 1
    
    # print(f"[STATE DEBUG] junction={junction_id}")
    # print(f"  neighbors={neighboring_nodes}")
    # if neighboring_nodes:
    #     print(f"  neighbor_features={[neighbor for neighbor in neighboring_nodes]}")
    # print(f"  → state_vector={state_vector}")
    # print("-" * 80)

    return np.array(state_vector, dtype=np.float32)

def get_state(node_id):
    junction_id=node_id
    neighbors=NEIGHBOR_MAP.get(junction_id,None)
    return get_paper_based_state(node_id,junction_id,neighbors)

def get_distacne(vehID, net):
    try:
        current_edge = traci.vehicle.getRoadID(vehID)
        nextNodeID = net.getEdge(current_edge).getToNode().getID()
        vehicle_pos = traci.vehicle.getPosition(vehID)
        junction_pos = traci.junction.getPosition(nextNodeID)
        junction_vehicle_distance = traci.simulation.getDistance2D(
            vehicle_pos[0], vehicle_pos[1], junction_pos[0], junction_pos[1])
        return junction_vehicle_distance
    except:
        pass

current_virtual_phases = {node: None for node in JUNCTION_NODES}
virtual_phase_start_times = {node: 0 for node in JUNCTION_NODES}

def traffic_control(nodeID, action, prev_t_start_dict):
    """
    action = 0 → 東西青（EW）
    action = 1 → 南北青（NS）
    信号切替時に3秒黄色＋1秒赤を追加
    """
    control_obj = {}
    prev_action = current_virtual_phases.get(nodeID, None)
    t_start = traci.simulation.getTime()

    # フェーズ切替がある場合のみ黄・赤信号を実施
    if prev_action is not None and prev_action != action:
        # 黄色3秒間：全車両減速
        yellow_duration = 3
        red_duration = 1
        end_yellow = t_start + yellow_duration
        end_red = end_yellow + red_duration

        #print(f"[DEBUG] node={nodeID} → 黄色信号 {yellow_duration}s, 赤信号 {red_duration}s")

        # 黄色信号（減速）
        while traci.simulation.getTime() < end_yellow:
            traci.simulationStep()
            for edge_obj in net.getNode(nodeID).getIncoming():
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_obj.getID()):
                    traci.vehicle.setSpeed(veh_id, 1.0)  # 減速
            time.sleep(0.05)

        # 赤信号（完全停止）
        while traci.simulation.getTime() < end_red:
            traci.simulationStep()
            for edge_obj in net.getNode(nodeID).getIncoming():
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_obj.getID()):
                    traci.vehicle.setSpeed(veh_id, 0.0)
            time.sleep(0.05)

    # --- 青信号フェーズ ---
    if action == 0:
        allowed_edges = EW_EDGES[nodeID]
    else:
        allowed_edges = NS_EDGES[nodeID]

    # 対応方向は青、他方向は赤
    for edge_obj in net.getNode(nodeID).getIncoming():
        edge_id = edge_obj.getID()
        for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
            if edge_id in allowed_edges:
                traci.vehicle.setSpeed(veh_id, SPEED)
                traci.vehicle.setColor(veh_id, (0, 255, 0))  # 緑
            else:
                traci.vehicle.setSpeed(veh_id, 0)
                traci.vehicle.setColor(veh_id, (255, 0, 0))  # 赤

    current_virtual_phases[nodeID] = action
    virtual_phase_start_times[nodeID] = traci.simulation.getTime()
    prev_t_start_dict[nodeID] = traci.simulation.getTime()

    return prev_t_start_dict


def get_reward(nodeID, prev_deadlocks_dict, ta=10.0):
    """
    論文「分布型マルチエージェント強化学習による幹線道路交差点」準拠の報酬関数。
    各方向の遅延(Delay_i)に到着率比 w_i を掛けて平均し、負値を報酬とする。
    
    Parameters
    ----------
    nodeID : str
        対象交差点のID（例："J1"）
    prev_deadlocks_dict : dict
        各ノードの前回デッドロック状態を保持する辞書
    ta : float
        行動間隔 (秒)。論文では10秒を使用。
    
    Returns
    -------
    reward : float
        負の加重遅延値（クリップ済み）
    prev_deadlocks_dict : dict
        更新後のデッドロック状態辞書
    """

    # --- 進入方向のEdge IDを取得 ---
    inbound_edges = [edge.getID() for edge in net.getNode(nodeID).getIncoming()]
    while len(inbound_edges) < 4:
        inbound_edges.append(None)

    # --- デッドロック判定 ---
    deadlock_detected = len(traci.simulation.getCollisions()) > 0

    delays = []
    counts = []

    for edge_id in inbound_edges:
        if edge_id is None:
            delays.append(0.0)
            counts.append(0.0)
            continue

        # --- 遅延時間（待ち時間） ---
        try:
            delay = traci.edge.getWaitingTime(edge_id)
        except traci.TraCIException:
            delay = 0.0

        # --- 車両数（到着率比の代用） ---
        try:
            num_veh = len(traci.edge.getLastStepVehicleIDs(edge_id))
        except traci.TraCIException:
            num_veh = 0

        delays.append(float(delay))
        counts.append(float(num_veh))

    # --- 到着率重み w_i を算出 ---
    total = sum(counts)
    if total > 0:
        weights = [c / total for c in counts]
    else:
        weights = [1.0 / len(delays)] * len(delays)

    # --- 重み付き平均遅延 ---
    weighted_delay = sum(w * d for w, d in zip(weights, delays))
    #print(f"[DEBUG REWARD] node={nodeID}, delays={delays}, weights={weights}, weighted_delay={weighted_delay:.2f}")

    # --- 報酬（負の加重遅延 / ta）---
    reward = - (weighted_delay / ta)

    # --- デッドロック補正 ---
    if deadlock_detected:
        reward -= 70
    elif prev_deadlocks_dict[nodeID] and not deadlock_detected:
        reward += 80

    # --- 安定化のためのクリッピング ---
    reward = np.clip(reward, -150, 150)

    # --- 状態を更新 ---
    prev_deadlocks_dict[nodeID] = deadlock_detected

    return reward, prev_deadlocks_dict


def set_simulation_time_limit(limit):
    global SIMULATION_TIME_LIMIT
    SIMULATION_TIME_LIMIT = limit

def reward_save_individual(episode, node_id, reward_value):
    individual_filename = f"reward_{name}_{node_id}.txt"
    individual_file = os.path.join("output", individual_filename)
    with open(individual_file, "a", encoding="utf-8") as f:
        f.write(f"{episode}\t{reward_value}\n")

def get_avg_waiting_time(nodeID):
    junction_edges = [edge_obj.getID() for edge_obj in net.getNode(nodeID).getIncoming()]
    waiting_times = [traci.edge.getWaitingTime(edge_id) for edge_id in junction_edges]
    return np.mean(waiting_times) if len(waiting_times) > 0 else 0

sync_interval = 5  # ターゲットネットワーク同期間隔（エピソードごと）
JUNCTION_NODES = [JUNCTION_NODE1, JUNCTION_NODE2, JUNCTION_NODE3]

def simulation(num, episode_num):
    # Shared replay buffer を作成してからエージェント群を初期化
    shared_replay = SharedReplayBuffer(capacity=20000, batch_size=32)

    t_start_per_junction = {node: 0 for node in JUNCTION_NODES}
    deadlock_history = []
    reward_history = {node: [] for node in JUNCTION_NODES}
    avg_waiting_time_history = {node: [] for node in JUNCTION_NODES}
    step_waiting_times = {node: [] for node in JUNCTION_NODES}
    SIGNAL_INTERVAL = 10

    # エピソード開始前に dummy_state を確定して各エージェントの state_size を決定
    # ここでは episode==1 の時に初期化を行う (後続は重み同期)
    central_trainer = None
    agents = {}

    for episode in range(1, episode_num+1):
        try:
            traci.close(False)
            time.sleep(0.3)
        except:
            pass
        init(False)
        # print("=== ReplayBuffer共有確認 ===")
        # print(f"SharedReplayBuffer のオブジェクトID: {id(shared_replay)}")

        # for node, agent in agents.items():
        #     print(f"{node} の replay_buffer ID: {id(agent.replay_buffer)}")

        # if central_trainer is not None:
        #     print(f"CentralTrainer の replay_buffer ID: {id(central_trainer.replay_buffer)}")
        # else:
        #     print("CentralTrainer はまだ初期化されていません。")
        # print("=================================")

        t_start_per_junction = {node: 0 for node in JUNCTION_NODES}
        if episode == 1:
            dummy_state = get_state(JUNCTION_NODE1)
            state_size = len(dummy_state)
            action_size = 2
            # 代表トレーナーを作り shared_replay を渡す
            central_trainer = DQNAgent(state_size, action_size, shared_replay)
            # 各交差点エージェントは同アーキテクチャで作る（パラメータ同期可能）
            agents = {
                JUNCTION_NODE1: DQNAgent(state_size, action_size, shared_replay),
                JUNCTION_NODE2: DQNAgent(state_size, action_size, shared_replay),
                JUNCTION_NODE3: DQNAgent(state_size, action_size, shared_replay)
            }
            # 初期的に central の重みを全員に配布
            for node, ag in agents.items():
                ag.qnet.load_state_dict(central_trainer.qnet.state_dict())
                ag.qnet_target.load_state_dict(central_trainer.qnet_target.state_dict())

        current_states = {node: get_state(node) for node in JUNCTION_NODES}
        dones = {node: False for node in JUNCTION_NODES}
        total_rewards = {node: 0 for node in JUNCTION_NODES}
        prev_deadlocks = {node: False for node in JUNCTION_NODES}
        epsilon = 0.01 + 0.9 * math.exp(-1. * episode / 200)
        teleported_vehicles = []
        deadlocks = 0
        step_waiting_times = {node: [] for node in JUNCTION_NODES}

        for i in range(num):
            make_vehicle(f"vehicle_{i}", make_random_route(i), 0)

        current_simulation_time = 0

        # シミュレーションループ
        while not all(dones.values()) and traci.simulation.getMinExpectedNumber() > 0 and SIMULATION_TIME_LIMIT > current_simulation_time:
            traci.simulationStep()
            current_simulation_time = traci.simulation.getTime()

            for vehID in teleported_vehicles:
                if vehID in traci.vehicle.getIDList():
                    traci.vehicle.setSpeed(vehID, SPEED)
            teleported_vehicles = traci.simulation.getEndingTeleportIDList()

            # 各交差点は独立して行動（実行時: 分散）
            for node_id in JUNCTION_NODES:
                agent = agents[node_id]
                state = current_states[node_id]
                elapsed = current_simulation_time - t_start_per_junction[node_id]
                if elapsed >= SIGNAL_INTERVAL or t_start_per_junction[node_id] == 0:
                    action = agent.get_action(state, epsilon)
                    t_start_per_junction = traffic_control(node_id, action, t_start_per_junction)

                    reward, prev_deadlocks = get_reward(node_id, prev_deadlocks,ta=10)
                    next_state = get_state(node_id)

                    # CTDE: 経験は shared replay に追加（node_id を付与）
                    shared_replay.add(node_id, state, action, reward, next_state, dones[node_id])

                    current_states[node_id] = next_state
                    total_rewards[node_id] += reward
                    avg_waiting = get_avg_waiting_time(node_id)
                    step_waiting_times[node_id].append(avg_waiting)

                    #print(f"[DEBUG] t={current_simulation_time:.1f}s, node={node_id} → action={action}, reward={reward:.3f}")

                if prev_deadlocks[node_id]:
                    deadlocks += 1

                if traci.simulation.getMinExpectedNumber() == 0 or current_simulation_time >= SIMULATION_TIME_LIMIT:
                    dones[node_id] = True

            # --- 学習（中央）: 各ステップで central_trainer が shared buffer からサンプルして更新 ---
            central_trainer.update_from_shared()

        # エピソード末の処理: ターゲットネット同期（エピソードごと）
        if episode % sync_interval == 0:
            central_trainer.sync_qnet()

        # 学習が進んだら central の重みを全エージェントへ配布
        for node, ag in agents.items():
            ag.qnet.load_state_dict(central_trainer.qnet.state_dict())
            ag.qnet_target.load_state_dict(central_trainer.qnet_target.state_dict())

        print(f"Episode: {episode}")
        for node_id in JUNCTION_NODES:
            reward_history[node_id].append(total_rewards[node_id])
            print("episode :{}, total reward : {}".format(episode, total_rewards[node_id]))
            reward_save_individual(episode, node_id, total_rewards[node_id])
            episode_avg_waiting = np.mean(step_waiting_times[node_id]) if step_waiting_times[node_id] else 0
            avg_waiting_time_history[node_id].append(episode_avg_waiting)
        deadlock_history.append(deadlocks)
        traci.close()

        # --- 学習後のモデル保存 ---
    os.makedirs("output", exist_ok=True)
    central_trainer.save("output/dqn_model_22_central.pth")
    for node_id, ag in agents.items():
        ag.save(f"output/dqn_model_22_{node_id}.pth")
    print("モデルを output フォルダに保存しました。")

    return reward_history, deadlock_history, avg_waiting_time_history

def test_simulation(num, episode_num):
    t_start_per_junction = {node: 0 for node in JUNCTION_NODES}
    SIGNAL_INTERVAL = 10
    junction_edges_map = {}
    for node_id in JUNCTION_NODES:
        edges = []
        for edge_obj in net.getNode(node_id).getIncoming():
            edges.append(edge_obj.getID())
        junction_edges_map[node_id] = edges

    for episode in range(1, episode_num + 1):
        init(True)
        current_states = {node: get_state(node) for node in JUNCTION_NODES}
        dones = {node: False for node in JUNCTION_NODES}
        total_rewards = {node: 0 for node in JUNCTION_NODES}
        prev_deadlocks = {node: False for node in JUNCTION_NODES}
        teleported_vehicles = []
        current_simulation_time = 0

        for i in range(num):
            make_vehicle(f"vehicle_{i}", make_random_route(i), 0)

        while not all(dones.values()) and traci.simulation.getMinExpectedNumber() > 0 and SIMULATION_TIME_LIMIT > current_simulation_time:
            traci.simulationStep()
            current_simulation_time = traci.simulation.getTime()

            for vehID in teleported_vehicles:
                if vehID in traci.vehicle.getIDList():
                    traci.vehicle.setSpeed(vehID, SPEED)
            teleported_vehicles = traci.simulation.getEndingTeleportIDList()

            for node_id in JUNCTION_NODES:
                agent = agents[node_id]
                state = current_states[node_id]
                elapsed = current_simulation_time - t_start_per_junction[node_id]
                if elapsed >= SIGNAL_INTERVAL or t_start_per_junction[node_id] == 0:
                    action = agent.get_action(state, epsilon=0.0)
                    t_start_per_junction = traffic_control(node_id, action, t_start_per_junction)

                    reward, prev_deadlocks = get_reward(node_id, prev_deadlocks)
                    next_state = get_state(node_id)
                    current_states[node_id] = next_state
                    total_rewards[node_id] += reward

                if traci.simulation.getMinExpectedNumber() == 0 or current_simulation_time >= SIMULATION_TIME_LIMIT:
                    dones[node_id] = True

        print(f"[Test] Episode {episode}")
        for node_id in JUNCTION_NODES:
            print(f"  Junction {node_id} total reward: {total_rewards[node_id]}")
        traci.close()

mode = input("Mode (train/test) :").strip().lower()
num_of_vehicles = int(input("Num of vehicles :"))
num_of_episode = int(input("Num of episode :"))
name = input("what is rewardfilename :")
limit = 3600
set_simulation_time_limit(limit)

if mode == "train":
    # 学習時: 新たに CTDE 版 simulation を実行
    if os.path.exists("output/dqn_model_22_J4.pth"):
        os.remove("output/dqn_model_22_J4.pth")
    if os.path.exists("output/dqn_model_22_J1.pth"):
        os.remove("output/dqn_model_22_J1.pth")
    if os.path.exists("output/dqn_model_22_J7.pth"):
        os.remove("output/dqn_model_22_J7.pth")

    reward_history, deadlock_history, avg_waiting_time_history = simulation(num_of_vehicles, num_of_episode)

    # 学習済み central_trainer の重みを各 node 用ファイルとして保存
    # central_trainer は simulation 内で作られるためここでは再読み込みして保存する形にするか
    # 簡便のため、直前の central_trainer をファイルスコープで持たせていない場合は
    # 学習後に各エージェントのネットワークを保存する（agents は simulation の外だと未定義なので再-init）
    # ここでは簡便に非GUIで1回起動して state_size を得て、モデルを保存する手順を reuse する
    try:
        # エピソード後に central_trainer の最後の重みで保存するため、
        # 再実行せずに簡潔に保存するために、上の simulation が返した情報をもとに
        # ここでは central_trainer のオブジェクト参照がまだ残っていればそれを使う想定にしています。
        # 実行環境によっては central_trainer をモジュールグローバルにする等の修正が必要です。
        # 安全策として、学習後にもう一度 init して状態を読み、agents を組んで central_trainer の重みを読み込み保存する
        init(False)
        dummy_state = get_state(JUNCTION_NODE1)
        state_size = len(dummy_state)
        traci.close()
        # 再作成して学習済みファイルをロードする設計にしていないため、
        # ユーザが望むなら「学習後に central model をファイル保存するコード」をsimulation 内に埋め込みします。
        print("Note: models saved inside simulation function in this CTDE script (or request explicit saving).")
    except:
        pass

elif mode == "test":
    try:
        init(False)
        dummy_state = get_state(JUNCTION_NODE1)
        state_size = len(dummy_state)
        traci.close()

        # --- エージェント作成（テスト用） ---
        agents = {
            JUNCTION_NODE1: DQNAgent(state_size),
            JUNCTION_NODE2: DQNAgent(state_size),
            JUNCTION_NODE3: DQNAgent(state_size)
        }

        agents[JUNCTION_NODE1].load("output/dqn_model_22_J4.pth")
        agents[JUNCTION_NODE2].load("output/dqn_model_22_J1.pth")
        agents[JUNCTION_NODE3].load("output/dqn_model_22_J7.pth")
        print("Models loaded.")
        test_simulation(num_of_vehicles, num_of_episode)
    except FileNotFoundError:
        print("Error: Model files not found. Please train the agents first.")
        sys.exit(1)

# ---------- 可視化 (元コードと互換) ----------
plt.figure()
for node_id, rewards in reward_history.items():
    plt.plot(range(1, len(rewards)+1), rewards, label=f"Junction {node_id}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Reward per Episode")
plt.show()

plt.figure()
plt.plot(range(1, len(deadlock_history)+1), deadlock_history)
plt.xlabel("Episode")
plt.ylabel("Deadlocks per Episode")
plt.title("Deadlocks per Episode")
plt.show()

plt.figure()
for node_id, waiting_times in avg_waiting_time_history.items():
    plt.plot(range(1, len(waiting_times)+1), waiting_times, label=f"Junction {node_id}")
plt.xlabel("Episode")
plt.ylabel("Average Waiting Time (s)")
plt.title("Average Waiting Time per Episode")
plt.legend()
plt.show()

plt.figure()
for node_id, agent in agents.items():
    plt.plot(agent.loss_history, label=f"Junction {node_id}")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("DQN Loss per Training Step")
plt.legend()
plt.show()