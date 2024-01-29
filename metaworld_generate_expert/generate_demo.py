# An example of generating expert demos by MetaWorld's scripted policies.
# Warning: The scripted policies may generate failure trajectories. You can write a task-dependent criterion to filter the generated trajectories.

import sys
sys.path.append("../IL")
import metaworld
import random
import metaworld.policies as policies
import cv2
import numpy as np

import pickle
from pathlib import Path
from collections import deque

from video import VideoRecorder

env_names = ["hammer-v2"]
num_demos = 10

POLICY = {
	'hammer-v2': policies.SawyerHammerV2Policy,
	'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
	'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
	'door-open-v2': policies.SawyerDoorOpenV2Policy,
	'bin-picking-v2': policies.SawyerBinPickingV2Policy,
	'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
	'door-unlock-v2': policies.SawyerDoorUnlockV2Policy,
	'basketball-v3': policies.SawyerBasketballV2Policy,
	'plate-slide-v2': policies.SawyerPlateSlideV2Policy,
	"hand-insert-v2": policies.SawyerHandInsertV2Policy,  
	"peg-insert-side-v2": policies.SawyerPegInsertionSideV2Policy,  
	'assembly-v3': policies.SawyerAssemblyV2Policy,
	'push-wall-v2': policies.SawyerPushWallV2Policy,
	'soccer-v2': policies.SawyerSoccerV2Policy,
	'disassemble-v2': policies.SawyerDisassembleV2Policy,
	'pick-place-wall-v3': policies.SawyerPickPlaceWallV2Policy,
	'pick-place-v2': policies.SawyerPickPlaceV2Policy,
	'push-wall-v2': policies.SawyerPushWallV2Policy,
	'lever-pull-v2': policies.SawyerLeverPullV2Policy,
	'stick-pull-v2': policies.SawyerStickPullV2Policy,
	'stick-pull-v2': policies.SawyerStickPullV2Policy,
	'shelf-place-v2': policies.SawyerShelfPlaceV2Policy,
	'window-close-v2': policies.SawyerWindowCloseV2Policy,
	'reach-v2': policies.SawyerReachV2Policy,
	'button-press-wall-v2': policies.SawyerButtonPressWallV2Policy,
	'box-close-v2': policies.SawyerBoxCloseV2Policy,
	'stick-push-v2': policies.SawyerStickPushV2Policy,
	'handle-pull-v2': policies.SawyerHandlePullV2Policy,
	'door-lock-v2': policies.SawyerDoorLockV2Policy,
}

CAMERA = {
	'hammer-v2': 'corner3',
	'drawer-close-v2': 'corner',
	'drawer-open-v2': 'corner',
	'door-open-v2': 'corner3',
	'bin-picking-v2': 'corner',
	'button-press-topdown-v2': 'corner',
	'door-unlock-v2': 'corner',
	'basketball-v3': 'corner',
	'plate-slide-v2': 'corner',
	'hand-insert-v2': 'corner',
	'peg-insert-side-v2': 'corner3',
	'assembly-v3': 'corner',
	'push-wall-v2': 'corner',
	'soccer-v2': 'corner',
	'disassemble-v2': 'corner',
	'pick-place-wall-v3': 'corner3',
	'pick-place-v2': 'corner3',
	'push-wall-v2': 'corner',
	'lever-pull-v2': 'corner4',
	'stick-pull-v2': 'corner3',
	'shelf-place-v2': 'corner',
	'window-close-v2': 'corner3',
	'reach-v2': 'corner3',
	'button-press-wall-v2': 'corner',
	'box-close-v2': 'corner3',
	'stick-push-v2': 'corner',
	'handle-pull-v2': 'corner3',
	'door-lock-v2': 'corner',
}

NUM_STEPS = {
	'hammer-v2': 125,
	'drawer-close-v2': 125,
	'drawer-open-v2': 125,
	'door-open-v2': 125,
	'bin-picking-v2': 175,
	'button-press-topdown-v2': 125,
	'door-unlock-v2': 125,
	'basketball-v3': 175,
	'plate-slide-v2': 125,
	'hand-insert-v2': 125,
	'peg-insert-side-v2': 150,
	'assembly-v3': 175,
	'push-wall-v2': 175,
	'soccer-v2': 125,
	'disassemble-v2': 125,
	'pick-place-wall-v3': 175,
	'pick-place-v2': 125,
	'push-wall-v2': 175,
	'lever-pull-v2': 125,
	'stick-pull-v2': 175,
	'shelf-place-v2': 175,
	'window-close-v2': 125,
	'reach-v2': 125,
	'button-press-wall-v2': 125,
	'box-close-v2': 175,
	'stick-push-v2': 125,
	'handle-pull-v2': 175,
	'door-lock-v2': 125,
}


for env_name in env_names:
	print(f"Generating demo for: {env_name}")
	# Initialize policy
	policy = POLICY[env_name]()

	# Initialize env
	mt1 = metaworld.MT1(env_name) # Construct the benchmark, sampling tasks
	# ml1 = metaworld.ML1(env_name)
	env = mt1.train_classes[env_name]()  # Create an environment with task `pick_place`

	# Initialize save dir
	save_dir = Path("./expert_demos") / env_name
	save_dir.mkdir(parents=True, exist_ok=True)

	# Initialize video recorder
	video_recorder = VideoRecorder(save_dir, camera_name=CAMERA[env_name])

	images_list = list()
	large_images_list = list()
	observations_list = list()
	actions_list = list()
	rewards_list = list()

	count = 0
	episode = 0
	while episode < num_demos:
		video_recorder.init(env)
		print(f"Episode {episode}")
		images = list()
		large_images = list()
		observations = list()
		actions = list()
		rewards = list()
		image_stack = deque([], maxlen=3)
		large_image_stack = deque([], maxlen=3)
		goal_achieved = 0

		# Set random goal
		task = mt1.train_tasks[count] #random.choice(ml1.train_tasks)
		print(count)
		count += 1
		env.set_task(task)  # Set task

		# Reset env
		observation = env.reset()  # Reset environment
		# video_recorder.record(env)
		num_steps = NUM_STEPS[env_name]
		for step in range(num_steps):
			# Get frames
			pixel = env.render(offscreen=True, camera_name=CAMERA[env_name])
			frame = cv2.resize(pixel.copy(), (84,84))
			frame = np.transpose(frame, (2,0,1))
			image_stack.append(frame)
			while(len(image_stack)<3):
				image_stack.append(frame)
			images.append(np.concatenate(image_stack, axis=0))
			large_frame = cv2.resize(pixel.copy(), (224,224))
			large_frame = np.transpose(large_frame, (2,0,1))
			large_image_stack.append(large_frame)
			while(len(large_image_stack)<3):
				large_image_stack.append(large_frame)
			large_images.append(np.concatenate(large_image_stack, axis=0))
			# Get action
			action = policy.get_action(observation)
			action = np.clip(action, -1.0, 1.0)
			actions.append(action)
			# Get observation
			observation[-3:] = 0
			observations.append(observation)
			# Act in the environment
			observation, reward, done, info = env.step(action)
			rewards.append(reward)
			video_recorder.record(env)
			goal_achieved += info['success'] 

		print(rewards[-1], np.max(rewards))
		# You can write a task-dependent criterion to filter the generated trajectories!!!
		if np.max(rewards) < 10:
			continue
		# Store trajectory
		episode = episode + 1
		images_list.append(np.array(images))
		large_images_list.append(np.array(large_images))
		observations_list.append(np.array(observations))
		actions_list.append(np.array(actions))
		rewards_list.append(np.array(rewards))

		video_recorder.save(f'demo{episode}.mp4')

	file_path = save_dir / 'expert_demos.pkl'
	payload = [images_list, observations_list, actions_list, rewards_list, large_images_list]


	with open(str(file_path), 'wb') as f:
		pickle.dump(payload, f)

