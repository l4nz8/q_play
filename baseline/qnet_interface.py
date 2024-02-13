import itertools
from pyboy import WindowEvent

class GameState():
    def __init__(self, pyboy):
        game_wrapper = pyboy.game_wrapper()

        # Calculate Mario's real X position on the level
        level_block = pyboy.get_memory_value(0xC0AB) # Level block offset
        mario_x = pyboy.get_memory_value(0xC202) # Mario's X position relative to screen
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0] # Screen scroll position
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16 # Adjust for tilemap alignment

        self.real_x_pos = level_block * 16 + real + mario_x # Compute Mario's absolute X position
        self.time_left = game_wrapper.time_left # Time remaining in the current level
        self.lives_left = game_wrapper.lives_left # Lives remaining
        self.score = game_wrapper.score # Current score
        self._level_progress_max = max(game_wrapper._level_progress_max, self.real_x_pos) # Track furthest level progress
        self.world = game_wrapper.world # Current world and level


class MarioAI():
	def __init__(self):
		self.realMax = [] # Track maximum progress in each level

	def GetReward(self, prevGameState: GameState, pyboy):
		"""
        Calculate the reward based on the change in game state.

		Args:
		- previousMario = Mario before step is taken
		- current_mario = Mario after step is taken
		"""
		timeRespawn = pyboy.get_memory_value(0xFFA6) # Check if Mario is in respawn state
		if(timeRespawn > 0):
			return 0 # No reward if Mario is respawning to avoid being punished for crossing a level
		
		current_mario = self.GetGameState(pyboy) # Get current game state

		# Check for level progression and reset max progress if a new level has been reached
		if max((current_mario.world[0] - prevGameState.world[0]), (current_mario.world[1] - prevGameState.world[1])):
			for _ in range(0,5):
				pyboy.tick() # Skip frames to ensure game state is updated
			current_mario = self.GetGameState(pyboy)

			# Update the recorded maximum progress
			pyboy.game_wrapper()._level_progress_max = current_mario.real_x_pos
			current_mario._level_progress_max = current_mario.real_x_pos

		# Initialize max level list with first level
		if len(self.realMax) == 0:
			self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])
		else:
			# Check if the current level's max progress is already tracked
			r = False
			for elem in self.realMax: #  Max length
				if elem[0] == current_mario.world[0] and elem[1] == current_mario.world[1]:
					elem[2] = current_mario._level_progress_max # Update max progress
					r = True
					break # leave loop
			
			# If this level is not already tracked, add it
			if r == False:
				self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])
			
		# Reward components
		clock = current_mario.time_left - prevGameState.time_left
		movement = current_mario.real_x_pos - prevGameState.real_x_pos
		death = -15*(current_mario.lives_left - prevGameState.lives_left)
		levelReward = 15*max((current_mario.world[0] - prevGameState.world[0]), (current_mario.world[1] - prevGameState.world[1])) # +15 if either new level or new world

		reward = clock + death + movement + levelReward
		return reward

	def GetActions(self):
		"""
        Generate a list of possible action combinations.
        """
		baseActions = [WindowEvent.PRESS_ARROW_RIGHT,
						WindowEvent.PRESS_BUTTON_A, 
						WindowEvent.PRESS_ARROW_LEFT]
		totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
		withoutRepeats = []
		for combination in totalActionsWithRepeats:
			reversedCombination = combination[::-1]
			if(reversedCombination not in withoutRepeats):
				withoutRepeats.append(combination)

		# Combine single and multi-action combinations
		filteredActions = [[action] for action in baseActions] + withoutRepeats
		del filteredActions[4] # Remove conflicting action ['PRESS_ARROW_RIGHT', 'PRESS_ARROW_LEFT']

		return filteredActions

	def PrintGameState(self, pyboy):
		"""
        Print the current game state for debugging.
        """
		gameState = GameState(pyboy)
		game_wrapper = pyboy.game_wrapper()

		print("'Fake', level_progress: ", game_wrapper.level_progress)
		print("'Real', level_progress: ", gameState.real_x_pos)
		print("_level_progress_max: ", gameState._level_progress_max)
		print("World: ", gameState.world)
		print("Time until respawn:", pyboy.get_memory_value(0xFFA6))

	def GetGameState(self, pyboy):
		"""
        Create a GameState instance representing the current state.
        """
		return GameState(pyboy)

	def GetLength(self, pyboy):
		"""
        Calculate the total length progressed across all levels.
        """
		result = sum([x[2] for x in self.realMax])
		pyboy.game_wrapper()._level_progress_max = 0 # Reset max level progress because game hasnt implemented it
		self.realMax = []
		return result