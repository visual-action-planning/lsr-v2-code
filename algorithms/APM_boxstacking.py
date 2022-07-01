from algorithms import APM_algorithm
import numpy as np
import os
import matplotlib as mpl
if not "DISPLAY" in os.environ:
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2


# ---
# ====================== Training functions ====================== #
# ---
class APM_boxstacking(APM_algorithm):
    def __init__(self, opt):
        super().__init__(opt)
        self.action_labels = ['loss', 'inputX', 'inputY', 'inputH', 'outputX', 'outputY']
        self.input_idx = [1, 2]
        self.output_idx = [4, 5]
        self.model_loss_idx = 0
            
    def get_box_center_from_x_y(self, array):
        """
        Returns the center coordinates corresponding to the box where the APN
        prediction x and y are pointing to. It assumes top left corner = (0,0), 
        x increasing positively downwards and y towards the right.
        """
        x, y = array[1], array[0]
        cx_vec = [55,115,185] # wrt the image coordinates (x positive towards right)
        cy_vec = [87,140,190] # wrt the image coordinates (y positive towards down)
        cx = cx_vec[y]
        if x == 2:
            cy = 195
        else:
            cy = cy_vec[x]
        return (cx,cy)


    def plot_prediction(self, img1, img2, pred_coords_scaled, coords_scaled, 
                            split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        img1 = self.vae.decoder(img1)[0]
        img2 = self.vae.decoder(img2)[0]

        # Descale coords back to the original size
        pred_coords = self.descale_coords(pred_coords_scaled.detach())
        coords = self.descale_coords(coords_scaled)
        
        plt.figure(1)        
        for i in range(n_subplots):
            # Start state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+1)
            pred_pick_xy = self.get_box_center_from_x_y(pred_coords[i][:2])
            actual_pick_xy = self.get_box_center_from_x_y(coords[i][:2])
            state1_img = (img1[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img1 = cv2.circle(state1_img, tuple(pred_pick_xy), 10, (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            
            # Start state predicted height for the robot and ground truth
            pred_pick_height = round(pred_coords_scaled[i][2].detach().item())
            plt.title('State 1, h_pred {0}'.format(pred_pick_height))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)            
            pred_place_xy = self.get_box_center_from_x_y(pred_coords[i][3:])
            actual_place_xy = self.get_box_center_from_x_y(coords[i][3:])
            
            state2_img = (img2[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img2 = cv2.circle(state2_img, tuple(pred_place_xy), 10, (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img2)
            plt.title('State 2')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        
        if new_save_path: 
            plt.savefig(new_save_path)
        else:
            plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
        plt.clf()
        plt.close('all')


    def monitor_epoch_training(self):
        self.plot_model_loss()
        self.plot_learning_curve()
        self.plot_epoch_time()            
        