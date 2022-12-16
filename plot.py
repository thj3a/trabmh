import matplotlib.pyplot as plt
import numpy as np
import os 

class Plot:

    def __init__(self):
        pass

    @classmethod
    def draw_plots(self, environment):
        if not environment.generate_plots:
            return

        self.textstr = '\n'.join((
            r'Encoding method= %s' %(environment.encoding,),
            r'Selection method= %s' %(environment.selection_method,),
            r'Mutation method= %s' %(environment.mutation_method,),
            r'Crossover method= %s' %(environment.crossover_method,),
            r'Stopping criteria= %s' %(environment.stop_message,),
            ))

        self.plot_time_to_best_sol(environment)
        self.plot_best_sol_tracking(environment)

    @classmethod
    def plot_time_to_best_sol(self, environment):
        times= np.array(environment.best_sol_change_times) - environment.start_time
        sols = np.unique(environment.best_sol_tracking)
        plt.plot(times, sols, color='tab:blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Best solution")
        plt.title("Exp. {} - Time to best solution found".format(str(environment.experiment_id)))  
        plt.axhline(y=environment.best_known_result, color='tab:red', linestyle='-')
        
        # place a text box in bottom right in axes coords
        ax = plt.gca()
        ax.text(0.4, 0.05, self.textstr, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', alpha=0.4))

        file_name = "{}_time_to_best_sol_.png".format(str(environment.experiment_id))
        plots_file = os.path.join(environment.plots_dir, file_name)
        plt.savefig(plots_file)
        plt.close()

    @classmethod
    def plot_best_sol_tracking(self, environment):
        plt.plot([i for i in range(len(environment.best_sol_tracking))], environment.best_sol_tracking, color='tab:blue')
        plt.xlabel("Generation")
        plt.ylabel("Best solution")
        plt.title("Exp. {} - Evolution of Best solution found so far".format(str(environment.experiment_id)))  
        plt.axhline(y=environment.best_known_result, color='tab:red', linestyle='-')
        
        # place a text box in bottom right in axes coords
        ax = plt.gca()
        ax.text(0.4, 0.05, self.textstr, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', alpha=0.4))


        file_name = "{}_best_sol_tracking_.png".format(str(environment.experiment_id))
        plots_file = os.path.join(environment.plots_dir, file_name)
        plt.savefig(plots_file)
        plt.close()
