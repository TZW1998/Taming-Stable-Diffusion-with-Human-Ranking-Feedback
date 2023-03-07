import numpy as np
import matplotlib.pyplot as plt

class RankSGD:
    
    def __init__(self, start_code,start_image, stepsize, smoothing_para):
        self.rounds = 1
        self.stepsize = stepsize
        self.smoothing_para = smoothing_para
        
        self.dim = start_code.shape
        
        self.best_code = start_code.flatten()
        self.best_image = start_image
        
        self.test_codes = None
        self.display_images = None
        
        self.mode = "grad_est" # switch between grad_est or line_search mode
        self.shuffled_ind = None
        self.search_direction = np.zeros_like(self.best_code)
        self.grad_accumulate_step = 0
        
        self.prev_best_image = None
        self.prev_best_code = None
    
    def generate_query_codes(self, num_query):
        if self.mode == "grad_est":
            test_d = np.random.randn(num_query, *self.best_code.shape) * self.smoothing_para 
        else:
            assert num_query > 2
            test_d = np.tile(np.expand_dims(self.search_direction, axis=0),(num_query-2,1))
            test_d *= np.array([scale for scale in (0.5 ** np.arange(num_query-2))]).reshape(num_query-2,1)
            test_d *= self.stepsize 
            
        self.test_codes = np.expand_dims(self.best_code, axis=0) + test_d 
            
        return self.test_codes.reshape(-1,*self.dim)
        
    def display_shuffled_images(self, generated_images, generation_time, maximum_display_rows = 6, plot = True):
        
        self.display_images = generated_images
        if (self.mode == "line_search"):
            self.display_images.append(self.prev_best_image)
            self.display_images.append(self.best_image)
            self.test_codes = np.concatenate([self.test_codes, np.expand_dims(self.prev_best_code,axis=0)])
            self.test_codes = np.concatenate([self.test_codes, np.expand_dims(self.best_code,axis=0)])
            
        if plot:
            self.shuffled_ind = np.random.permutation(len(self.test_codes)).tolist()
        else:
            print(self.shuffled_ind)
            self.shuffled_ind = np.arange(len(generated_images))
        
        if plot:
            nrows = len(self.display_images) // maximum_display_rows + 1
            if len(self.display_images) % maximum_display_rows == 0:
                nrows -= 1
            ncols = min([maximum_display_rows,len(self.display_images)])

            fig, ax = plt.subplots(nrows, ncols, figsize=(12*nrows,6*ncols),dpi=500, constrained_layout=True)

            if nrows > 1:
                for i in range(nrows):
                    for j in range(ncols):

                        nq = i * ncols + j
                        ax[i][j].axis('off')
                        if nq < len(self.shuffled_ind):
                            t_ind = self.shuffled_ind[nq]
                            ax[i][j].imshow(self.display_images[t_ind])
                            fig_id = "ID:{}".format(nq+1)
                            # mark the  previous best code
                            if (t_ind == (len(self.display_images) - 1)) and (self.mode == "line_search"):
                                fig_id += "*"
                            elif (t_ind == (len(self.display_images) - 2)) and (self.mode == "line_search"):
                                fig_id += "**"
                            ax[i][j].set_title(fig_id)
                        else:
                            fig.delaxes(ax[i][j])
            else:
                for nq in range(ncols):
                    ax[nq].axis('off')
                    if nq < len(self.shuffled_ind):
                        t_ind = self.shuffled_ind[nq]
                        ax[nq].imshow(self.display_images[t_ind])
                        fig_id = "ID:{}".format(nq+1)
                        # mark the previous best code
                        if (t_ind == (len(self.display_images) - 1)) and (self.mode == "line_search"):
                            fig_id += "*"
                        elif (t_ind == (len(self.display_images) - 2)) and (self.mode == "line_search"):
                            fig_id += "**"
                        ax[nq].set_title(fig_id)
            plt.show()
            

        print(f"\033[1;32m Current Round: {self.rounds}, Generation time: {generation_time} secs \n")
        #plt.savefig(save_path + f"/process{self.rounds}.png",bbox_inches="tight",dpi=500)
        
    
    def rank_feedback(self, rank_info):
        print(self.mode, self.grad_accumulate_step)
        self.rounds += 1
        if self.mode == "grad_est":
            # using the rank information to compute the gradient
            rank_info = [int(r) for r in rank_info]
            test_codes_rank = {}
            for t in range(len(self.test_codes)):
                if (t+1) in rank_info:
                    test_codes_rank[t] = rank_info.index(t+1)
                else:
                    test_codes_rank[t] = -1

            #rank-based update
            update_direction = np.zeros_like(self.best_code)
            accumulated_weights = 0

            # print(test_codes_rank)
            for tc, tr in test_codes_rank.items():
                if tr>= 0:
                    update_direction += (len(self.test_codes)-2*tr) * (self.test_codes[tc] - self.best_code)
                else:
                    update_direction += (- len(rank_info)) * (self.test_codes[tc] - self.best_code)

            k=len(rank_info)
            m=len(self.test_codes)
            update_direction /= k*(k-1)/2 + k*(m-k)
            
        
            self.search_direction = self.search_direction * self.grad_accumulate_step + update_direction
            self.grad_accumulate_step += 1
            self.search_direction /= self.grad_accumulate_step
            
            self.mode = "line_search"
            
            best_ind = int(rank_info[0])-1
            self.prev_best_code = self.test_codes[best_ind]
            if self.display_images is not None:
                self.prev_best_image = self.display_images[best_ind]
            
        else:
            if self.shuffled_ind is not None:
                best_ind = self.shuffled_ind[int(rank_info[0])-1]
            else:
                best_ind = int(rank_info[0])-1
            print("best",rank_info,best_ind+1)
            if best_ind != (len(self.test_codes) - 1):
                #print(np.linalg.norm(self.test_codes[best_ind]-self.best_code))
                print("found better solution")
                self.best_code = self.test_codes[best_ind]
                self.grad_accumulate_step = 0
                self.search_direction = np.zeros_like(self.best_code)
                if self.display_images is not None:
                    self.best_image = self.display_images[best_ind]
                    
            self.mode = "grad_est"