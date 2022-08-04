import random
import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
import time
import progress.bar as Bar

start = time.time() #user time
t = time.process_time() # process time

#Nivestar

global_best_execution = 0
global_nTimes_got_optimal_solution = 0

CODE_VERBOSE = True
GRAPH = True
DEBUG = False
MINIMIZATION = True #flag to change to maximization (False) or minimization (True)

# Clonal Selection for travelling salesman problem
class Clonal_Selection():

	def __init__(self, N = 200, n = 100, d= 0.8, beta = 1, n_generations = 50, rho = 10, 
			tournament_count = 5, selection1='T', selection2='R'):

		self.N = N # antibody population size
		self.n = n #selection area size
		self.d = round(N*d) #amount of antibody to be replaced
		self.beta = beta # cloning factor
		self.n_generations = n_generations

		assert (n <= N), "Error, the antibody population size cannot be larger than selection area size"
		assert (d <= 1), "Error, the amount of antibody to be replaced cannot be bigger than selection area size"
		if(self.d < tournament_count):
			print(f"warning: Tournament Count was readjusted to {self.d} values")
			tournament_count = self.d
		
		self.tournament_count = tournament_count
		self.rho = rho #the mutation rate weight
		self.first_selection = selection1
		self.second_selection = selection2

		self.antibodies = [] 
		self.M = [] # memory group

		self.file_reading()

		count_cities = len(self.distance_matrix)
		self.L  = count_cities

	def file_reading(self):

		self.distance_matrix, self.solution = read_file()

	def run(self):
	
		global global_nTimes_got_optimal_solution, 	global_best_execution

		generation = 1
		if CODE_VERBOSE:
			print("Creating a population of ", self.N, " antibodies")
		bests = []
		bests_ind = []

		self.antibodies = self.create_antibodies( Pr= self.N ) # on first iteration P = N

		while (generation < self.n_generations):
			
			affinities = self.evaluation(self.antibodies)

			if DEBUG:
				print("Generation ", generation)
				print("Antibodies: \n", self.antibodies[:])
				print("Scores:\n", affinities[:])
			
			best = min(affinities)
			bests.append(best)
			bests_ind.append( self.antibodies[affinities.index(best)] )

			# first selection
			if(self.first_selection == 'T'):
				selected = self.selection_tournament( self.antibodies, self.n)
			elif(self.first_selection == 'R'):
				selected = self.selection_roulette(self.antibodies, self.n) 
			else:
				print("Error selection method")
				exit()

			#cloning
			C = self.cloning(selected)
			C_mature = self.hyper_mutation(C)

			#second selection
			if(self.second_selection == 'T'):
				self.selection_tournament(C_mature, self.d)
			elif(self.second_selection == 'R'):
				selected = self.selection_roulette(C_mature, self.d) 
			else:
				print("Error selection method")
				exit()

			new_M = self.selection_tournament(C_mature, self.d) 
			self.replacement(new_M)
			Pr = (self.N - self.d)
			antibodies = self.create_antibodies(Pr) # renovating antibodies
			self.antibodies = antibodies + self.M
			generation+=1

		affinities = self.evaluation(self.antibodies)
		if CODE_VERBOSE:
			print("END")		
			print("Final Antibody Population:\n", self.antibodies)
			print("Final Score Population:\n", affinities)
		
		best = min(affinities)
		bests.append(best)
		bests_ind.append( self.antibodies[affinities.index(best)] )
		if DEBUG:	
			print("Bests Scores: ", bests)
		temp = bests.copy()
		if MINIMIZATION:
			temp.sort()
		else:
			temp.sort(reverse=True)
		best = temp[0]
		best_index = bests.index(best)

		print(f"The individual: { bests_ind[best_index] } was the best, with score: { self.func_obj(bests_ind[best_index]) }") 
		print(f"Solution:\t{self.solution}, Solution score: {self.func_obj(self.solution)}")

		if(self.func_obj(self.solution) == best):
			print("\nBest Solution Reached")
			global_nTimes_got_optimal_solution +=1
		global_best_execution = best

		if GRAPH:
			self.draw_graph(bests, best, best_index, self.func_obj(self.solution))

	def create_antibodies(self, Pr):

		antibodies = []
		for _ in range (Pr):
			antibodies.append(self.create_chromossomes())	
		return antibodies

	def create_chromossomes(self): #can change with the problem

		new_born = [*range(1, self.L +1, 1)]
		random.shuffle(new_born)
		return new_born 

	def evaluation(self, values):

		affinities = [] #clean old evaluations
		for ind in values:
			affinities.append(self.func_obj(ind))
		return affinities

	def func_obj(self, antibody):

		distance = 0
		for c, _ in enumerate(antibody):
			if (c < self.L-1):
				distance += self.distance_matrix[antibody[c]-1][antibody[c+1]-1]
		distance += self.distance_matrix[antibody[self.L-1]-1][antibody[0]-1]
		return distance

	def selection_roulette(self, population, count):

		selected = []
		roulette_values = []

		affinities = self.evaluation(population)

		for antibody in range (len(population)):

			if(MINIMIZATION):
				affinity_r = 1/affinities[antibody]   
			else:
				affinity_r = affinities[antibody]  
			roulette_values.append(affinity_r) #probability 

		sum_r = sum(roulette_values)
		selection_probs = [ c/sum_r for c in roulette_values] 
		for _ in range(count):  
			index_sorted_antibody = np.random.choice(len(affinities), p=selection_probs)
			sorted = population[index_sorted_antibody]
			selected.append(sorted)
		return selected 

	def selection_tournament(self, population, count):

		winners = []
		for _ in range (count):
			competitors = random.sample(population, self.tournament_count)
			competitors_affi = self.evaluation(competitors)

			if MINIMIZATION:
				winner_aff = min(competitors_affi)			
			else:
				winner_aff  = max(competitors_affi)

			winner_index = competitors_affi.index(winner_aff)
			winner = competitors[winner_index]
			winners.append(winner)

		return winners

	def cloning(self , selected):

		C = [] #intermediate population 
		for antibody in selected:
	
			affinity = self.func_obj(antibody)
			N_clones = 1 + round(self.beta * self.N / affinity) #makes clonning based in score and self.beta
			for _ in range(N_clones):
				new_clone = antibody.copy()
				C.append(new_clone)
		return C

	def hyper_mutation(self , C):  #make mutation change for his affinities
		
		clone_affinities = self.evaluation(C)

		for clone in C:
			index = random.randrange(0, len(clone)-1)
			if MINIMIZATION:
				affinity_normalization =  1 - self.func_obj(clone)/max(clone_affinities) #afinity minimization
			else:
				affinity_normalization =  self.func_obj(clone)/max(clone_affinities)
			
			mutation_rate = math.exp(-self.rho*affinity_normalization)
			self.switch_mutation(clone, index, mutation_rate)

		return C

	def switch_mutation(self, clone, index, mutation_rate):

		if random.uniform(0, 1) < mutation_rate:
			sorted = random.randrange(1, self.L)
			while(sorted == index):
				sorted = random.randrange(1, self.L)
			switcher = clone[sorted]
			clone[sorted] = clone[index]
			clone[index] = switcher

	def replacement(self, new_M):

		self.M = new_M 

	def draw_graph(self, bests, best, best_index, solution_score):

		plt.figure(figsize=(16,8))
		plt.plot( [*range(self.n_generations)], bests)
		plt.axline( (0, solution_score), (self.n_generations-3, solution_score), color='g', linestyle='--', label="Best Solution")
		plt.annotate(f'{solution_score}', xy=(self.n_generations-2, solution_score), fontsize=10)
		plt.annotate(f'{best}', xy=(best_index, best+0.01*(best)), xytext=(best_index, best+0.05*(bests[0])), arrowprops=dict(facecolor='black', shrink=0.01, headwidth=10, headlength=10,width=1), fontsize=10)
		plt.ylabel("Distance")
		plt.xlabel("Generations")
		plt.xticks([*range(self.n_generations)])
		plt.legend(loc='lower left')
		plt.show() 

def read_file():

	f_distance_matrix = np.loadtxt("tests/lau15_dist.txt", dtype='int')

	n_fsolution= "tests/lau15_tsp.txt"
	f_solution = open(n_fsolution)
	solution = f_solution.readlines()
	solution = [int(s) for s in solution]
	
	return f_distance_matrix, solution

def calibrate():
	
	global global_best_execution, global_nTimes_got_optimal_solution, DEBUG, CODE_VERBOSE, GRAPH

	CODE_VERBOSE = False
	DEBUG = False
	GRAPH = False

	#params
	selection1_list = ['T', 'R'] 
	selection2_list = ['R', 'T']
	N_list = [200]
	n_list = [100, 50] 
	d_list = [0.8, 0.5] 
	beta_list = [1, 10]
	rho_list = [1, 10]

	df = pandas.DataFrame()
	selection1_column = []
	selection2_column = []
	N_column = []
	n_column = []
	d_column = []
	beta_column = []
	rho_column = []
	params_columns = []

	list_best_execution = [] #best result for execution 
	times_best_solution = 0

	final_mean_list_best_execution = [] #mean of best reasults for execution 
	final_sum_times_best_solution = []
	times_repetition = 10

	total_progress = len(selection1_list) * len(selection2_list) *	len(N_list) * len(n_list ) \
		* len(d_list ) * len(beta_list ) * len(rho_list ) * times_repetition
	my_bar = Bar.ShadyBar('Calibrating...', max=total_progress,  suffix='%(percent)d%%')

	
	for s1 in selection1_list:
		for s2 in selection2_list:
			for N in N_list:
				for n in n_list:
					for d in d_list:
						for b in beta_list:
							for r in rho_list:
								selection1_column.append(s1)
								selection2_column.append(s2)
								N_column.append(N)
								n_column.append(n)
								d_column.append(d)
								beta_column.append(b)
								rho_column.append(r)		
								print("Execution with the current params: ")
								print(f'N: {N} n: {n} s1: {s1} s2:{s2} d:{d} rho:{r} bheta:{b}')
								params_columns.append(f'N: {N} n: {n} s1: {s1} s2:{s2} d:{d} rho:{r} bheta:{b}')

								for _ in range(times_repetition):
									print('\n\n')
									my_bar.next()
									print('\n\n')

									CS = Clonal_Selection(N = N, n =n, d= d, beta = b, rho =r, n_generations = 60, 
										tournament_count= 10, selection1 = s1, selection2 = s2)
									CS.run( )

									list_best_execution.append(global_best_execution)
									times_best_solution += global_nTimes_got_optimal_solution

									#resets
									global_best_execution = []
									global_nTimes_got_optimal_solution = 0

				
								final_mean_list_best_execution.append(np.mean(list_best_execution))
								final_sum_times_best_solution.append(times_best_solution)
								
								#clean 
								list_best_execution.clear()
								times_best_solution = 0

	df['Params'] = params_columns
	df['N'] = N_column						
	df["n"] = n_column 
	df["d"] = d_column
	df["Selection1"] = selection2_column 
	df['Selection2'] = selection2_column
	df['Rho'] = rho_column
	df['Beta'] = beta_column
	df["Mean_bests_Results"] = final_mean_list_best_execution
	df["Number_times_got_best_solution"] = final_sum_times_best_solution
	
	df.to_csv('results_params_ClonalSelection.csv', sep=';')

if __name__ == "__main__":

	CS = Clonal_Selection(N = 200, n = 100, d= 0.8, beta = 1, rho = 5, n_generations = 60, 
		tournament_count= 10, selection1 = 'T', selection2 = 'R')
	CS.run()
	#calibrate() # only for calibrate

	end = time.time() #user
	user_time = end - start 
	elapsed_time = time.process_time() - t #process

	print("="*100) 
	print("User time: %s" % user_time)
	print("Process time: %s" % elapsed_time)
	print( time.strftime("%H hours %M minutes %S seconds", time.gmtime(user_time)) ) 
	print("="*100)
