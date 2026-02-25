

total_exec = 0
success_exec = 0

scene_name = "FloorPlan" + str(floor_no)

c = Controller( height=1000, width=1000)
c.reset(scene_name)
no_robot = len(robots)

# initialize n agents into the scene
multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.25, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=no_robot))

# MAP-THOR: scene pre-initialization and checker setup
# task_folder is injected by execute_plan.py (e.g. "1_put_bread_lettuce_tomato_fridge")
if task_folder is not None:
    _scene_init_mod, _checker_mod = get_scene_initializer(task_folder, scene_name, ignore_applicable=False)
else:
    _scene_init_mod, _checker_mod = None, None

checker = _checker_mod.Checker() if _checker_mod is not None else None

# MAP-THOR: run preinit to place objects at task-specific initial positions
if _scene_init_mod is not None:
    try:
        c.last_event = _scene_init_mod.SceneInitializer().preinit(c.last_event, c)
        print(f"[MAP-THOR] preinit complete: {task_folder}/{scene_name}")
    except Exception as _e:
        print(f"[MAP-THOR] preinit failed (continuing): {_e}")

# MAP-THOR: register all scene objects with checker (enables per-object counting)
if checker is not None:
    _all_oids = [obj["objectId"] for obj in c.last_event.metadata["objects"]]
    if hasattr(checker, "all_objects"):
        checker.all_objects(obj_ids=_all_oids, scene=scene_name)

# add a top view camera
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# get reachable positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

# randomize positions of the agents
for i in range (no_robot):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))

objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])

# MAP-THOR checker uses numerated object names (e.g., Bread_1).
# Build a stable map from objectId -> numerated name for this episode.
_checker_obj_name = {}
_name_counts = {}
for _obj in c.last_event.metadata["objects"]:
    _oid = _obj["objectId"]
    _nm = _oid.split("|")[0]
    _name_counts[_nm] = _name_counts.get(_nm, 0) + 1
    _checker_obj_name[_oid] = f"{_nm}_{_name_counts[_nm]}"

action_queue = []

task_over = False

recp_id = None

# Capture script directory before the exec_actions thread starts.
# __file__ may be unavailable in thread context in some Python versions.
try:
    _exec_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _exec_dir = os.path.abspath(os.getcwd())

# per-robot inventory: stores the objectType name currently held ("" if empty)
inventory = [""] * no_robot

for i in range (no_robot):
    multi_agent_event = c.step(action="LookDown", degrees=35, agentId=i)


def _obj_readable(object_id):
    """Convert AI2-THOR objectId to readable type name.
    e.g. 'Apple|+00.12|+01.22|-00.34' -> 'Apple'
    """
    return object_id.split("|")[0] if object_id else ""


def _obj_checker_name(object_id):
    """Return numerated object name (e.g. Bread_1) expected by MAP-THOR checker."""
    if not object_id:
        return ""
    return _checker_obj_name.get(object_id, _obj_readable(object_id))


def exec_actions():
    global total_exec, success_exec, inventory
    # delete if current output already exist
    cur_path = _exec_dir + "/*/"
    for x in glob(cur_path, recursive = True):
        shutil.rmtree (x)

    # create new folders to save the images from the agents
    for i in range(no_robot):
        folder_name = "agent_" + str(i+1)
        folder_path = _exec_dir + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # create folder to store the top view images
    folder_name = "top_view"
    folder_path = _exec_dir + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_counter = 0

    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)

                elif act['action'] == 'MoveAhead':
                    multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])

                elif act['action'] == 'MoveBack':
                    multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])

                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'PickupObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                        inventory[act['agent_id']] = _obj_checker_name(act['objectId'])
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    # MAP-THOR checker: report PickupObject (also auto-credits NavigateTo)
                    if checker is not None:
                        _readable = _obj_checker_name(act['objectId'])
                        checker.perform_metric_check(
                            f"PickupObject({_readable})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'PutObject':
                    total_exec += 1
                    # Capture inventory before put (checker needs it as the "object in inventory" arg)
                    _inv_before_put = inventory[act['agent_id']]
                    multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                        inventory[act['agent_id']] = ""
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    # MAP-THOR checker: report PutObject(receptacle, carried_object)
                    if checker is not None:
                        _recp_readable = _obj_checker_name(act['objectId'])
                        # BaseChecker expects receptacle in action string and carried object
                        # via inventory_object argument (not embedded in action string).
                        _action_str = f"PutObject({_recp_readable})"
                        checker.perform_metric_check(_action_str, _success, _inv_before_put)

                elif act['action'] == 'ToggleObjectOn':
                    total_exec += 1
                    multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"ToggleObjectOn({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'ToggleObjectOff':
                    total_exec += 1
                    multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"ToggleObjectOff({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'OpenObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="OpenObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"OpenObject({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'CloseObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="CloseObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"CloseObject({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'SliceObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="SliceObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"SliceObject({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'CleanObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="CleanObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"CleanObject({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'ThrowObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])

                elif act['action'] == 'BreakObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="BreakObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    _success = multi_agent_event.metadata['errorMessage'] == ""
                    if _success:
                        success_exec += 1
                    else:
                        print(multi_agent_event.metadata['errorMessage'])
                    if checker is not None:
                        checker.perform_metric_check(
                            f"BreakObject({_obj_checker_name(act['objectId'])})", _success, inventory[act['agent_id']]
                        )

                elif act['action'] == 'Done':
                    multi_agent_event = c.step(action="Done")


            except Exception as e:
                print (e)

            for i,e in enumerate(multi_agent_event.events):
                cv2.imshow('agent%s' % i, e.cv2img)
                f_name = _exec_dir + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
                cv2.imwrite(f_name, e.cv2img)
            top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
            cv2.imshow('Top View', top_view_rgb)
            f_name = _exec_dir + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, top_view_rgb)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            img_counter += 1
            action_queue.pop(0)

actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()

def GoToObject(robots, dest_obj):
    global recp_id

    # check if robots is a list

    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)

    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    dest_obj_id = None
    dest_obj_center = None
    if "|" in dest_obj:
        # obj alredy given
        dest_obj_id = dest_obj
        pos_arr = dest_obj_id.split("|")
        dest_obj_center = {'x': float(pos_arr[1]), 'y': float(pos_arr[2]), 'z': float(pos_arr[3])}
    else:
        # Common category aliases that may appear in LLM-generated plans.
        alias_patterns = {
            "Cutlery": ["Fork", "Spoon", "ButterKnife", "Knife", "Spatula"],
        }
        candidate_patterns = [dest_obj]
        if dest_obj in alias_patterns:
            candidate_patterns = alias_patterns[dest_obj]

        for pattern in candidate_patterns:
            for idx, obj in enumerate(objs):
                match = re.match(pattern, obj)
                if match is not None:
                    dest_obj_id = obj
                    dest_obj_center = objs_center[idx]
                    if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                        break  # find the first reachable instance
            if dest_obj_id is not None:
                break

    if dest_obj_id is None or dest_obj_center is None:
        print(f"[GoToObject] WARNING: target '{dest_obj}' not found in scene objects, skipping.")
        return False

    print ("Going to ", dest_obj_id, dest_obj_center)

    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

    # closest reachable position for each robot
    # all robots cannot reach the same spot
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

    goal_thresh = 0.25
    # at least one robot is far away from the goal

    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1

            # get the pose of robot
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}

            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])

            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move
                count_since_update[ia] += 1
            else:
                # robot moving
                count_since_update[ia] = 0

            if count_since_update[ia] < 8:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

            time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}

    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']

    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})

    print ("Reached: ", dest_obj)
    if dest_obj == "Cabinet" or dest_obj == "Fridge" or dest_obj == "CounterTop":
        recp_id = dest_obj_id
    return True

def PickupObject(robots, pick_obj):
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal
    for idx in range(no_agents):
        robot = robots[idx]
        print ("PIcking: ", pick_obj)
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        # list of objects in the scene and their centers
        objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

        for idx, obj in enumerate(objs):
            match = re.match(pick_obj, obj)
            if match is not None:
                pick_obj_id = obj
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break # find the first instance
        # GoToObject(robot, pick_obj_id)
        # time.sleep(1)
        print ("Picking Up ", pick_obj_id, dest_obj_center)
        action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        time.sleep(1)

def PutObject(robot, put_obj, recp):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])

    metadata = c.last_event.events[agent_id].metadata
    robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    dist_to_recp = 9999999 # distance b/w robot and the recp obj
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            dist = objs_dists[idx]
            if dist < dist_to_recp:
                recp_obj_id = obj
                dest_obj_center = objs_center[idx]
                dist_to_recp = dist


    global recp_id
    # if recp_id is not None:
    #     recp_obj_id = recp_id
    # GoToObject(robot, recp_obj_id)
    # time.sleep(1)
    action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
    time.sleep(1)

def PutObjectInFridge(robot, put_obj, recp):
    """Compatibility wrapper for plans that use explicit fridge placement action."""
    PutObject(robot, put_obj, recp)

def SwitchOn(robot, sw_obj):
    print ("Switching On: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    # turn on all stove burner
    if sw_obj == "StoveKnob":
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                GoToObject(robot, sw_obj_id)
                # time.sleep(1)
                action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.1)

    # all objects apart from Stove Burner
    else:
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        GoToObject(robot, sw_obj_id)
        time.sleep(1)
        action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)

def SwitchOff(robot, sw_obj):
    print ("Switching Off: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    # turn on all stove burner
    if sw_obj == "StoveKnob":
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.1)

    # all objects apart from Stove Burner
    else:
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        GoToObject(robot, sw_obj_id)
        time.sleep(1)
        action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)

def OpenObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance

    global recp_id
    if recp_id is not None:
        sw_obj_id = recp_id

    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)

def CloseObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance

    global recp_id
    if recp_id is not None:
        sw_obj_id = recp_id

    GoToObject(robot, sw_obj_id)
    time.sleep(1)

    action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id})

    if recp_id is not None:
        recp_id = None
    time.sleep(1)

def BreakObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)

def SliceObject(robot, sw_obj):
    print ("Slicing: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)

def CleanObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)

def ThrowObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance

    action_queue.append({'action':'ThrowObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)
