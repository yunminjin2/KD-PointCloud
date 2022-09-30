from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import torch

scale = 1.0

def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss
    
def loss_fn_kd_2(outputs, fps_idxs, gt_flow, teacher_outputs, teacher_fps_idxs, gamma, alpha=[0.02, 0.04, 0.08, 0.16]):
    # student - teacher
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs)
    KD_loss += gamma*loss1 + (1-gamma)*loss2
    
    return KD_loss

def attentiveImitationLoss(outputs, fps_idxs, gt_flow, teacher_outputs, teacher_fps_idxs, t_history, gamma, alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss =torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss_ST = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs)
    loss_SG = multiScaleLoss(outputs, gt_flow, fps_idxs)
    loss_TG = multiScaleLoss(teacher_outputs, gt_flow, teacher_fps_idxs)


    sigma = 1 - ((loss_TG)/(max(t_history) - min(t_history)))

    KD_loss += gamma*(loss_SG) + (1-gamma)*sigma*(loss_ST)

    return KD_loss

def biDirectionLoss(outputs, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, teacher_fps_idxs, gamma1, gamma2, beta, alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)

    g_loss1 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    g_loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs2)

    k_loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    k_loss2 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs2)


    KD_loss += beta*(gamma1*k_loss1 + (1-gamma1)*g_loss1) + (1-beta)*(gamma2*k_loss2 + (1-gamma2)*g_loss2)
    return KD_loss


def loss_fn_ht(outputs, feat1s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, teacher_fps_idxs, gamma, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)

    Hint_loss = ((feat1s[layer]-t_feat1s[layer])**2)/2


    KD_loss += gamma*loss1 + (1-gamma)*loss2  + Hint_loss.sum()/(feat1s[0].nelement())

    return KD_loss

def biDirection_loss_ht(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, t_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, beta, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)

    src_hint_loss = ((feat1s[layer]-t_feat1s[layer])**2)/2
    target_hint_loss = ((feat2s[layer]-t_feat2s[layer])**2)/2


    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(0.5*src_hint_loss.sum() + 0.5*target_hint_loss.sum())
    
    return KD_loss

def flow_loss_ht(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, t_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, beta, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    num_scale = len(outputs)
    offset = len(fps_idxs) - num_scale + 1

    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)
    

    loss2 = 0
    for i in range(len(outputs)):
        diff_flow = outputs[i].permute(0, 2, 1) - teacher_outputs[i].permute(0, 2, 1)
        loss2 += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
    
    src_hint_loss = ((feat1s[layer]-t_feat1s[layer])**2)/2
    target_hint_loss = ((feat2s[layer]-t_feat2s[layer])**2)/2

    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(0.5*src_hint_loss.sum() + 0.5*target_hint_loss.sum())
    return KD_loss


def att_iter_loss(outputs, c_feat1s, c_feat2s, fps_idxs1, fps_idxs2, gt_flow, t_outputs, t_c_feat1s, t_c_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, layers=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()

    loss1 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
        
    loss2 = 0
    gt_flows = [gt_flow]
    for i in range(1, len(t_fps_idxs1) + 1):
        fps_idxs = t_fps_idxs1[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idxs) / scale
        gt_flows.append(sub_gt_flow)

    distil_ratios = []
    for each_layer in layers:
        diffs = []
        for each_iter in range(len(t_outputs[each_layer])):
            diffs.append(((t_outputs[each_layer][each_iter].permute(0, 2, 1) - gt_flows[each_layer])**2).sum(dim=1).sum(dim=1))
        diffs = torch.stack(diffs, 1)
  
        distil_ratios.append(1 - softmax(diffs))
    distil_ratios = torch.stack(distil_ratios, 1).permute(2, 1, 0)

    src_ht = torch.zeros(1).cuda()
    # target_ht = torch.zeros(1).cuda()

    for i, each_layer in enumerate(layers):
        for each_iter in range(len(t_outputs[each_layer])):
            diff = torch.norm(outputs[each_layer].permute(0, 2, 1) -t_outputs[each_layer][each_iter].permute(0, 2, 1), dim = 2).sum(dim = 1)
            tmp = torch.t(distil_ratios[i][each_iter]) @ diff
            src_ht += alpha[each_layer] * tmp.mean()
    # loss2 = 0.5*(src_ht + target_ht)

    KD_loss += gamma*loss1 + (1-gamma)*src_ht
    
    return KD_loss

def att_ht_loss(outputs, c_feat1s, c_feat2s, fps_idxs1, fps_idxs2, gt_flow, t_outputs, t_c_feat1s, t_c_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, layers=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()

    loss1 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
        
    loss2 = 0
    gt_flows = [gt_flow]
    for i in range(1, len(t_fps_idxs1) + 1):
        fps_idxs = t_fps_idxs1[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idxs) / scale
        gt_flows.append(sub_gt_flow)

    distil_ratios = []
    for each_layer in layers:
        diffs = []
        for each_iter in range(len(t_outputs[each_layer])):
            diffs.append(((t_outputs[each_layer][each_iter].permute(0, 2, 1) - gt_flows[each_layer])**2).sum(dim=1).sum(dim=1))
        diffs = torch.stack(diffs, 1)
  
        distil_ratios.append(1 - softmax(diffs))
    distil_ratios = torch.stack(distil_ratios, 1).permute(2, 1, 0)

    src_ht = torch.zeros(1).cuda()
    target_ht = torch.zeros(1).cuda()

    for i, each_layer in enumerate(layers):
        for each_iter in range(len(t_outputs[each_layer])):
            diff_ht = torch.norm(((c_feat1s[each_layer] - t_c_feat1s[each_layer][each_iter])**2)/2 , dim=2).sum(dim=1)
            tmp = torch.t(distil_ratios[i][each_iter]) @ diff_ht
            src_ht += alpha[each_layer] * tmp.mean()
    # loss2 = 0.5*(src_ht + target_ht)

    KD_loss += gamma*loss1 + (1-gamma)*src_ht
    
    return KD_loss

def cross_biDirection_loss_ht(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, t_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, beta, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    
    src_hint_loss = torch.zeros(1).cuda()
    
    

    for each in layer:
        t_feats = torch.cat([t_feat1s[each], t_feat2s[each]], dim=1)
        src_hint_loss += ((feat1s[each]-t_feats)**2).sum()/2


    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(src_hint_loss )

    return KD_loss


def cross_loss(outputs, crosses, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_crosses, t_fps_idxs1, t_fps_idxs2, gamma, beta, alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)

    cross_loss = 0
    for layer in range(len(crosses)):
        cross_loss += ((((crosses[layer]-t_crosses[layer])**2)/2).sum())/crosses[layer].nelement()

    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(cross_loss)

    return KD_loss


def bridge_ht_loss(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, t_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, beta, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    
    src_hint_loss = ((feat1s[layer]-t_feat1s)**2)/2
    target_hint_loss = ((feat2s[layer]-t_feat2s)**2)/2


    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(0.5*src_hint_loss.sum() + 0.5*target_hint_loss.sum())

    return KD_loss

def bridge_ht_loss_iter(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, t_feat2s, t_fps_idxs1, t_fps_idxs2, gamma, beta, layer=[2, 3],  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    
    src_hint_loss = 0
    target_hint_loss = 0

    for i, each_layer in enumerate(layer):
        src_hint_loss += (((feat1s[each_layer]-t_feat1s[i])**2)/2).sum()
        target_hint_loss += (((feat2s[each_layer]-t_feat2s[i])**2)/2).sum()


    KD_loss += beta*(gamma*loss1 + (1-gamma)*loss2) + (1-beta)*(0.5*src_hint_loss + 0.5*target_hint_loss)

    return KD_loss



def double_bridge_ht_loss(outputs, feat1s, feat2s, fps_idxs1, fps_idxs2, crosses, gt_flow, teacher_outputs, br_feat1s, br_feat2s, t_fps_idxs1, t_fps_idxs2, br_crosses, gamma, beta, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    def crossLoss(crosses, br_crosses, alpha):
        cross_loss=0
        for layer in range(len(br_crosses)):
            cross_loss += ((((crosses[layer]-br_crosses[layer])**2)/2).sum())/crosses[layer].nelement()
        return cross_loss

    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    loss3 = crossLoss(crosses, br_crosses, alpha)

    src_hint_loss = ((feat1s[layer]-br_feat1s)**2)/2
    target_hint_loss = ((feat2s[layer]-br_feat2s)**2)/2

    KD_loss += beta*(gamma[0]*loss1 + gamma[1]*loss2 + gamma[2]*loss3) + (1-beta)*(0.5*src_hint_loss.sum() + 0.5*target_hint_loss.sum())

    return KD_loss