# Public libraries:
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def file_reader(path):
    char_map={'[':'', ' ':'', ']':'', '\n':''}
    with open(Path(path), "r") as file:
        out = []
        for line in file:
            l = list(filter(None, line.translate(str.maketrans(char_map)).split(';')))
            out.append([list(map(int, list(filter(None, e.split(','))))) for e in l])
    return out






def file_writer(results, path):
    with open(Path(path), "w") as file:
        for result in results:
            line = ""
            for task in result:
                e_str = ""
                for e in task: e_str += str(e) + ","
                line += f"[{e_str}];"
            file.write(line + "\n")

            
            
            
            
            
            
            
            

def save_figs(examples, results, title, path, time_limit = 40):
    # set length of deadline and periodic line. and create list of Ylabels
    d_len = 0.3
    Ylabels = []
    
    for result in results:
        
        # create figure for every line of results
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        plt.xlabel("Real-Time clock")
        plt.ylabel("Tasks")
        plt.xlim((0, time_limit))
        row = results.index(result)
        
        # seperating missed tasks from original results.
        missed = result[-1]
        
        # do processing for every task in each lines.
        for task in result[ : -1]:
            
            # create deadline and periodic lines.
            index = result.index(task)
            if index < len(examples[0][0]):
                Ylabels.append(' ')
                Ylabels.append('Task - '+ str(index + 1))

                # t is period of task and d is deadline of it.
                t = examples[0][0][index]
                d = examples[0][2][index]

                d = np.array(range(0, time_limit, t)) + d
                t = range(0, time_limit, t)
                d = list(zip(d, 0.1 * np.ones(shape = (len(d), 1))))
                t = list(zip(t, 0.1 * np.ones(shape = (len(t), 1))))
                plt.broken_barh(d, (2 * index - d_len, d_len), color='red')
                plt.broken_barh(t, (2 * index + 1, d_len), color='black')
            
                # create color bar of main tasks.
                missed_task = list(zip(range(len(task)), (np.array(task) * np.array(missed))))
                output = list(zip(range(len(task)), task))
                plt.broken_barh(output, (2 * index, 1))
                plt.broken_barh(missed_task, (2 * index, 1), color='yellow')
            
            # check for interrupt
            if index == len(examples[0][0]):
                output = list(zip(range(len(task)), task))
                plt.broken_barh(output, (2 * index, 1), color='red')
            plt.grid(True)
            
        ax.set_yticklabels(Ylabels)
        plt.show()

        
        
        
        
        
        
        
        
        

def rm_scheduler(examples, time_limit = 40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        missed = []
        result = [[] for i in range(len(T))]
        
        # your code goes here :: Start
        T, C, D = np.array((T, C, D))
        C_first = C.copy()
        isnt_done = np.ones(shape = D.shape)
        clocks = np.zeros(shape = D.shape)
        
        # main body
        C = np.zeros(shape = C.shape)
        time = 0
        step = 1
        while time < time_limit:
            
            # updating job's computational time when it's period is gone!
            C[time % T == 0] += C_first[time % T == 0]
            isnt_done[time % T == 0] = 1
            
            # searching for minimum period that dosent process yet to do its task.
            index = np.argmin(isnt_done * T)
            
            # define processing output
            Pr = np.zeros(shape = D.shape)
            
            # just process when there are some work to do.
            if C[index] > 0: 
                
                # check for missing job.
                if np.floor(clocks[index] / C_first[index]) * T[index] + D[index] <= time:
                    missed.append(1)
                else:
                    missed.append(0)
                    
                # do processing!
                C[index] -= step
                clocks[index] += 1
                
                # adding processing result to our result
                Pr[index] = 1
                for task in result:
                    task.append(Pr[result.index(task)])
                    
                # if a task is done it's time to set it has no computional load to do.
                if C[index] <= 0:
                    isnt_done[index] = np.Inf
                
            # if there is no work to do.
            else:
                for task in result:
                    task.append(Pr[result.index(task)])
                missed.append(0)
                
            # go to next step
            time += step
        
        # your code goes here :: End
        
        result.append(missed)
        results.append(result)
    return results










def dm_scheduler(examples, time_limit = 40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        missed = []
        result = [[] for i in range(len(T))]
        
        
        # your code goes here :: Start
        T, C, D = np.array((T, C, D))
        C_first = C.copy()
        isnt_done = np.ones(shape = D.shape)
        clocks = np.zeros(shape = D.shape)
        
        
        # main body
        C = np.zeros(shape = C.shape)
        time = 0
        step = 1
        while time < time_limit:
            
            # updating job's computational time when it's period is gone!
            C[time % T == 0] += C_first[time % T ==0]
            isnt_done[time % T == 0] = 1
            
            # searching for minimum period that dose not process yet to do task.
            index = np.argmin(isnt_done * D)
            
            # define processing output 
            Pr = np.zeros(shape = D.shape)
            
            # just process when there are some work to do.
            if C[index] > 0:
                
                # check for missing job.
                if np.floor(clocks[index] / C_first[index]) * T[index] + D[index] <= time:
                    missed.append(1)
                else:
                    missed.append(0)
                    
                # do processing!
                C[index] -= step
                clocks[index] += 1
                
                # adding processing result to our result 
                Pr[index] = 1
                for task in result:
                    task.append(Pr[result.index(task)])
                    
                    
                # if a task is done it's time to set it has no computation load to do.
                if C[index] <= 0:
                    isnt_done[index] = np.Inf
                    
            # if there is no work to do.
            else:
                for task in result:
                    task.append(Pr[result.index(task)])
                missed.append(0)
                
            # go to next step 
            time += step
            
        # your code goes here :: End
        result.append(missed)
        results.append(result)
    return results










def ed_scheduler(examples, time_limit = 40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        missed = []
        result = [[] for i in range(len(T))]
        # your code goes here :: Start ------------------------
        
        # convert our list to np.array and save first states.
        T, C, D = np.array((T, C, D))
        C_first = C.copy()
        D_first = D.copy()
        isnt_done = np.ones(shape = D.shape)
        clocks = np.zeros(shape = D.shape)
        
        # main body
        C = np.zeros(shape = C.shape)
        time = 0
        step = 1
        while time < time_limit:
            
            # updating job's C when its period is gone!
            C[time % T == 0] += C_first[time % T == 0] 
            isnt_done[time % T == 0] = 1
            
            # searching for closest deadline to do.
            index = np.argmin(isnt_done * D)
            
            # define processing output
            Pr = np.zeros(shape = D.shape)
            
            # Just process when there are some work to do.
            if C[index] > 0:
                
                # Check for missing job
                if np.floor(clocks[index] / C_first[index]) * T[index] + D_first[index] <= time:
                    missed.append(1)
                else:
                    missed.append(0)
                    
                # do process!    
                C[index] -= step
                clocks[index] += 1
                
                # updating deadlines. when previous task is done!
                if clocks[index] % C_first[index] == 0:
                    D[index] += T[index]
                
                # adding processing result to our result
                Pr[index] = 1
                for task in result:
                    task.append(Pr[result.index(task)])
                
                # if a task is done! it's time to update its deadline
                if C[index] <= 0:
                    isnt_done[index] = np.Inf
                    
            # if there arn't any work to do.        
            else:   
                for task in result:
                    task.append(Pr[result.index(task)])
                missed.append(0)
                
            # go to next step
            time += step 
            
        # your code goes here :: End ------------------------

        result.append(missed)
        results.append(result)
    return results











def ap_rm_scheduler(examples, ap_task_time, ap_task_jobs, time_limit = 40):
    # This is an Interrupt-Driven Aperiodic RM task scheduler
    # In this scheduler, Aperiodic task should be processed 
    # immediately after reception to the server.
    # Periodic Tasks may miss some deadlines.
    
    
    results = []
    
    
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        missed = []
        interupt = []
        result = [[] for i in range(len(T))]
        
        # your code goes here :: Start
        T, C, D = np.array((T, C, D))
        C_first = C.copy()
        isnt_done = np.ones(shape = D.shape)
        clocks = np.zeros(shape = D.shape)
        
        # main body
        C = np.zeros(shape = C.shape)
        time = 0
        step = 1
        interrupt_p = 0
        interrupt_l = 0
        while time < time_limit:
            
            # create random interrupts
            if interrupt_p == 0:
                interrupt_p = np.floor(1.1 * np.random.rand())
            if interrupt_p > 0 & interrupt_l == 0:
                interrupt_l = np.random.randint(4)
                    
            # updating job's computational time when it's period is gone!
            C[time % T == 0] += C_first[time % T == 0]
            isnt_done[time % T == 0] = 1
            
            # searching for minimum period that dosent process yet to do its task.
            index = np.argmin(isnt_done * T)
            
            # define processing output
            Pr = np.zeros(shape = D.shape)
            
            # just process when there are some work to do.
            if (C[index] > 0) & (interrupt_l == 0): 
                
                # check for missing job.
                if np.floor(clocks[index] / C_first[index]) * T[index] + D[index] <= time:
                    missed.append(1)
                else:
                    missed.append(0)
                    
                # do processing!
                C[index] -= step
                clocks[index] += 1
                
                # adding processing result to our result
                Pr[index] = 1
                for task in result:
                    task.append(Pr[result.index(task)])
                interupt.append(0)
                    
                # if a task is done it's time to set it has no computional load to do.
                if C[index] <= 0:
                    isnt_done[index] = np.Inf
                
            # if there is no work to do or interrupts occur.
            else:
                for task in result:
                    task.append(Pr[result.index(task)])
                missed.append(0)
                
                # check for interrupts.
                if interrupt_l:
                    interrupt_l -= 1
                    if not interrupt_l:
                        interrupt_p = 0
                    interupt.append(1) 
                
            # go to next step
            time += step
        
        # your code goes here :: End
        
        result.append(interupt)
        result.append(missed)
        results.append(result)
    return results













def rm_pooling_server_scheduler(examples, time_limit = 40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        missed = []
        result = [[] for i in range(len(T))]
        
        T, C, D = np.array((T, C, D))
        C_first = C.copy()
        isnt_done = np.ones(shape = T.shape)
        clocks = np.zeros(shape = D.shape)
        
        
        # main body
        C = np.zeros(shape = C.shape)
        time = 0
        step = 1
        while time < time_limit:
            
            # updating job's computation time when it's period is gone!
            C[time % T ==0] += C_first[time % T == 0]
            isnt_done[time % T == 0] =1 
            
            # create 3rd task at random time
            if time % T[2] == 0:
                aperiodic_p = np.floor(1.5 * np.random.rand())
                if aperiodic_p == 0:
                    C[2] -= C_first[2]
                    
            # searching for the highest proiority task that dose not process yet to do.
            index = np.argmin(isnt_done * T)
            
            # define processing output 
            Pr = np.zeros(shape = D.shape)
            
            # just process when there are some work to do.
            if C[index] > 0:
                
                # check for missing job.
                if np.floor(clocks[index] / C_first[index]) * T[index] + D[index] <= time:
                    missed.append(1)
                else:
                    missed.append(0)
                    
                # do processing 
                C[index] -= step
                clocks[index] += 1
                
                # adding processing result to our result
                Pr[index] = 1
                for task in result:
                    task.append(Pr[result.index(task)])
                    
                # if a task is done it's time to set it has no computation load to do.
                if C[index] <= 0:
                    isnt_done[index] = np.Inf
                    
            # if there is no work to do.
            else:
                for task in result:
                    task.append(Pr[result.index(task)])
                missed.append(0)
                
            # go to next step
            time += step
                
        result.append(missed)
        results.append(result)
        # END OF THE MAIN LOOP (FOR)
        
    # Showing result
    d_len = 0.3
    Ylabels = []
    
    for result in results:
        
        # create figure for every line of results
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        plt.xlabel("Real-Time clock")
        plt.ylabel("Tasks")
        plt.xlim((0, time_limit))
        row = results.index(result)
        
        # seperating missed tasks from original results.
        missed = result[-1]
        
        # do processing for every task in each lines.
        for task in result[ : -1]:
            
            # create deadline and periodic lines.
            index = result.index(task)
            Ylabels.append(' ')
            Ylabels.append('Task - '+ str(index + 1))
            
            # create color bar of main tasks.
            missed_task = list(zip(range(len(task)), (np.array(task) * np.array(missed))))
            output = list(zip(range(len(task)), task))
            
            # t is period of task and d is deadline of it.
            t = examples[0][0][index]
            d = examples[0][2][index]

            d = np.array(range(0, time_limit, t)) + d
            t = range(0, time_limit, t)
            d = list(zip(d, 0.1 * np.ones(shape = (len(d), 1))))
            t = list(zip(t, 0.1 * np.ones(shape = (len(t), 1))))
            plt.broken_barh(d, (2 * index - d_len, d_len), color='red')
            plt.broken_barh(t, (2 * index + 1, d_len), color='black')
            plt.broken_barh(missed_task, (2 * index, 1), color='yellow')
            
            
            if index != 2:
                plt.broken_barh(output, (2 * index, 1), color='green')
                
            
            # check for interrupt
            if index == 2:
                plt.broken_barh(output, (2 * index, 1), color='blue')
            plt.grid(True)
            
        ax.set_yticklabels(Ylabels)
        plt.show()
        
        
        
