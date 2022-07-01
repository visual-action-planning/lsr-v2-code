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
class APM_ropebox(APM_algorithm):
    def __init__(self, opt):
        super().__init__(opt)
        self.action_labels = ['loss', 'actionTypeloss', 'pickBoxXloss', 'pickBoxYloss', 'placeBoxXloss', 'placeBoxYloss']
        self.input_idx = [2, 3]
        self.output_idx = [4, 5]
        self.model_loss_idx = 0

    def get_box_center_from_x_y(self, array):
        """
        Returns the center coordinates corresponding to the box where the APM
        prediction x and y are pointing to. It assumes top left corner = (0,0),
        x increasing positively downwards and y towards the right.
        """
        action_type, pickX, pickY, placeX, placeY = array
        if action_type == 2:
            return (0, 0), (0, 0)
        else:
            pickID = pickX * 3 + pickY
            placeID = placeX * 3 + placeY
            positions = [(50, 75), (125, 75), (200, 75),
                            (50, 150), (125, 150), (200, 150),
                            (50, 210), (125, 210), (200, 210)]
            pick_coords = positions[pickID]
            place_coords =  positions[placeID]
            return pick_coords, place_coords


    def plot_actionType_loss(self):
        """Plots epochs vs model loss."""
        # Losses on the action type
        plt_data = np.stack(self.epoch_losses)
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 1], 'g-', linewidth=2, label='actionTypeloss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Box loss')
        plt.savefig(self.save_path + '_actionLoss')
        plt.close()
        
        
    def plot_prediction(self, img1, img2, pred_coords_scaled, coords_scaled,
                            split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        img1 = self.vae.decoder(img1)[0]
        img2 = self.vae.decoder(img2)[0]

        # Descale coords back to the original size
        pred_coords = self.descale_coords(pred_coords_scaled.detach())
        coords = self.descale_coords(coords_scaled)
        n_subplots = min(img1.shape[0], n_subplots)

        plt.figure(1)
        for i in range(n_subplots):
            # Start state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+1)

            pred_action_type = round(pred_coords[i][0])
            pred_pick, pred_place = self.get_box_center_from_x_y(pred_coords[i])
            actual_pick, actual_place = self.get_box_center_from_x_y(coords[i])

            state1_img = (img1[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img1 = cv2.circle(state1_img, tuple(pred_pick), 10, (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            plt.suptitle('action t{0}/p{1}; pick t{2}/p{3}; place t{4}/p{5}'.format(
                round(coords[i][0]), pred_action_type, actual_pick, pred_pick, actual_place, pred_place))

            plt.title('State 1')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)
            state2_img = (img2[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img2 = cv2.circle(state2_img, tuple(pred_place), 10, (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place), 15, (0, 255, 0), 4)
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
        self.plot_actionType_loss()
        self.plot_learning_curve()
        self.plot_epoch_time()