MULTI_CHIP = False
CHIP_NUM = 0

###### Supported Workloads #############
# Tiny-YOLOv2
#INPUT_FILE = "input/tiny_yolov2.txt"
INPUT_FILE = "input/tiny_yolov2_16x16.txt"
CUTPOINT = 4
WIDTH = 224
HEIGHT = 224

# YOLOv3 128x128
#INPUT_FILE = "input/yolov3_128_32x32.txt"
#CUTPOINT = 9
#WIDTH = 128
#HEIGHT = 128

# YOLOv3 416x416
#INPUT_FILE = "input/yolov3_416_32x32.txt"
#CUTPOINT = 9
#WIDTH = 416
#HEIGHT = 416

# EfficientnetS
#INPUT_FILE = "input/efficientnetS_32x32.txt"
##CUTPOINT = 11 # not yet verified for the shortcut across cutlayer
#CUTPOINT = 15 # default
#WIDTH = 224
#HEIGHT = 224

# Mobilenetv1
#INPUT_FILE = "input/mobilenet_v1_ssd.txt"
##CUTPOINT = 13 # not yet verified for the depthwise cutlayer
#CUTPOINT = 14 # default
#WIDTH = 224
#HEIGHT = 224

# ResNet50
#INPUT_FILE = "input/resnet50.txt"
#CUTPOINT = 14
#WIDTH = 224
#HEIGHT = 224
########################################

OUTPUT_FILE = "npu_cfg.c"
PF = 16 # Parallelism Factor (Ti=To)
DRAM_BASE_ADDR = 0xf0000000
CHANNEL = PF

# Buffer
buf_size = 0x12200
allocated = []

# Buffer1
NPU_MEM1_BASE_ADDR = 0x60000000
free_list1 = [(0, buf_size)]
max_addr1 = 0

# Buffer2
# NPU_MEM2_BASE_ADDR = 0x61000000
# [2023.10.09] To use continueous address space for Yolov2-tiny in Silterra memory model
NPU_MEM2_BASE_ADDR = 0x60012200
free_list2 = [(0, buf_size)]
max_addr2 = 0

# Buffer3
NPU_MEM3_BASE_ADDR = 0x62000000
free_list3 = [(0, buf_size)]
max_addr3 = 0

def allocate(data_size, layers, buf_idx):
    if buf_idx == 1:
        global max_addr1
        for i,(addr,size) in enumerate(free_list1):
            if data_size <= size:
                free_list1[i] = (addr+data_size, size-data_size)
                allocated.append((addr,data_size,layers,buf_idx))
                if addr+data_size > max_addr1:
                    max_addr1 = addr+data_size
                    #print("Max addr1 Updated to {} for layer{}".format(max_addr1,layers))

                return NPU_MEM1_BASE_ADDR+addr
        raise Exception("Buf1 Allocation error")
    elif buf_idx == 2:
        global max_addr2
        for i,(addr,size) in enumerate(free_list2):
            if data_size <= size:
                free_list2[i] = (addr+data_size, size-data_size)
                allocated.append((addr,data_size,layers,buf_idx))
                if addr+data_size > max_addr2:
                    max_addr2 = addr+data_size
                    #print("Max addr2 Updated to {} for layer{}".format(max_addr2,layers))

                return NPU_MEM2_BASE_ADDR+addr
        raise Exception("Buf2 Allocation error")
    elif buf_idx == 3:
        global max_addr3
        for i,(addr,size) in enumerate(free_list3):
            if data_size <= size:
                free_list3[i] = (addr+data_size, size-data_size)
                allocated.append((addr,data_size,layers,buf_idx))
                if addr+data_size > max_addr3:
                    max_addr3 = addr+data_size
                    #print("Max addr3 Updated to {} for layer{}".format(max_addr3,layers))

                return NPU_MEM3_BASE_ADDR+addr
        raise Exception("Buf3 Allocation error")


def search(layer_idx, num):
    # Iterate through allocated block list
    addr_list = []
    for i,(addr, data_size, layers, buf_idx) in enumerate(allocated):
        if layer_idx in layers:
            if buf_idx == 1:
                addr_list.append(NPU_MEM1_BASE_ADDR+addr)
            elif buf_idx == 2:
                addr_list.append(NPU_MEM2_BASE_ADDR+addr)
            elif buf_idx == 3:
                addr_list.append(NPU_MEM3_BASE_ADDR+addr)

            if num == 1:
                return addr_list
            else:
                num = num -1


def free(layer_idx):
    # Iterate through allocated block list
    for alloc_data in allocated[:]:
        data_addr, data_size, layers, buf_idx = alloc_data
        if layer_idx in layers:
            layers.remove(layer_idx)
            if len(layers) >= 1:
                continue
            else:
                # No more layers to use this data
                # Remove from allocated list, Update free_list
                allocated.remove(alloc_data)

                if buf_idx == 1:
                    for j,(addr,size) in enumerate(free_list1):
                        if addr > data_addr:
                            free_list1.insert(j,(data_addr,data_size))
                            coalesce(buf_idx)
                            break
                elif buf_idx == 2:
                    for j,(addr,size) in enumerate(free_list2):
                        if addr > data_addr:
                            free_list2.insert(j,(data_addr,data_size))
                            coalesce(buf_idx)
                            break
                elif buf_idx == 3:
                    for j,(addr,size) in enumerate(free_list3):
                        if addr > data_addr:
                            free_list3.insert(j,(data_addr,data_size))
                            coalesce(buf_idx)
                            break


def coalesce(buf_idx):
    if buf_idx == 1:
        success = 0
        for i,(addr,size) in enumerate(free_list1):
            if i < len(free_list1)-1 and addr+size == free_list1[i+1][0]:
                free_list1[i] = (addr, size+free_list1[i+1][1])
                del(free_list1[i+1])
                success = 1
        if success == 1:
            coalesce(buf_idx)
    elif buf_idx == 2:
        success = 0
        for i,(addr,size) in enumerate(free_list2):
            if i < len(free_list2)-1 and addr+size == free_list2[i+1][0]:
                free_list2[i] = (addr, size+free_list2[i+1][1])
                del(free_list2[i+1])
                success = 1
        if success == 1:
            coalesce(buf_idx)
    elif buf_idx == 3:
        success = 0
        for i,(addr,size) in enumerate(free_list3):
            if i < len(free_list3)-1 and addr+size == free_list3[i+1][0]:
                free_list3[i] = (addr, size+free_list3[i+1][1])
                del(free_list3[i+1])
                success = 1
        if success == 1:
            coalesce(buf_idx)

def parse_config_file(config_file_path):
    """
    Parses a YOLOv2 configuration file and returns a list of dictionaries with the parsed values.
    """
    config_list = []
    current_section = None
    with open(config_file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config_list.append({'type': current_section})
            elif line:
                key, value = line.split('=')
                if key == 'layers':
                    value = value.split(',')
                    try:
                        value = [int(v.strip()) for v in value]
                    except:
                        pass

                elif value.isnumeric():
                    value = int(value)
                config_list[-1][key] = value

    return config_list

if __name__ == '__main__':
    cfg = parse_config_file(INPUT_FILE)

    # Ignore [net]
    if cfg[0]['type'] == 'net':
        cfg = cfg[1:]

    # Initialization
    in_width = WIDTH
    in_height = HEIGHT
    in_ch = CHANNEL
    total_param_size = 0
    num_layer = 0

    # Output filename
    if MULTI_CHIP:
        name,ext = OUTPUT_FILE.split('.')
        OUTPUT_FILE = name + "_chip" + str(CHIP_NUM) + "." + ext
    npu_cfg = open(OUTPUT_FILE, 'w')

    # Preprocess
    for i,layer in enumerate(cfg):
        # init
        weight_size = 0
        scale_bias_size = 0
        param_size = 0
        size = 0

        if layer['type'] == 'convolutional':
            size = layer['size']
            stride = layer['stride']
            filters = layer['filters']
            pad = layer['pad']
            if 'depthwise' in layer: # jckim [2023.04.19]
                is_depthwise = layer['depthwise']
            else:
                is_depthwise = 0

            if(filters % PF != 0):
               new_filters = (filters//PF)*PF+PF
               print("[Warning] {} filter: {} -> {}".format(layer['name'],filters, new_filters))
               filters = new_filters

            out_width = (in_width - size + 2*pad)//stride + 1
            out_height = (in_height - size + 2*pad)//stride + 1
            out_ch = filters
            num_layer += 1

            # Calculate Param
            # TODO : 1st layer different
            if(is_depthwise==1): # jckim [2023.04.19]
                print("depthwise in_ch:",in_ch)
                weight_size = size*size*in_ch # weight(1B)
                out_ch = in_ch # route layer -> depthwise layer case
            else:
                weight_size = size*size*in_ch*out_ch # weight(1B)

            scale_bias_size = 2*out_ch + 2*out_ch # scale(2B) + bias(2B)
            param_size = weight_size + scale_bias_size
            total_param_size += param_size
            #print("{} {} {} {} {}".format(size, in_ch, out_ch, stride, in_width))

        elif layer['type'] == 'maxpool':
            size = layer['size']
            stride = layer['stride']
            out_width = (in_width - 1)//stride + 1 #TODO size=1 fixed?
            out_height = (in_height - 1)//stride + 1 #TODO size=1 fixed?
            out_ch = in_ch
            cfg[i-1]['out_size'] = out_width * out_height * out_ch # Update OFM size

        elif layer['type'] == 'upsample':
            stride = layer['stride']
            out_width = in_width * 2
            out_height = in_height * 2
            out_ch = in_ch
            cfg[i-1]['out_size'] = out_width * out_height * out_ch # Update OFM size

        elif layer['type'] == 'shortcut':
            from_idx = layer['from']
            from_layer = cfg[from_idx]
            if from_layer['type'] == 'convolutional':
                cfg[from_idx]['to_residual'] = num_layer-1
            elif from_layer['type'] == 'shortcut' or from_layer['type'] == 'maxpool':
                if cfg[from_idx-1]['type'] != 'convolutional':
                    raise Exception("???")
                cfg[from_idx-1]['to_residual'] = num_layer-1

        elif layer['type'] == 'route':
            route_indices = layer['layers']
            out_ch = 0
            for route_idx in route_indices:
                out_ch += cfg[route_idx]['out_ch']
            cfg[i+1]['route'] = route_indices
            if len(route_indices) == 2: # concat
                if cfg[i-1]['type'] == 'convolutional':
                    cfg[i-1]['out_size'] = cfg[i-1]['out_width'] * cfg[i-1]['out_height'] * out_ch
                elif cfg[i-2]['type'] == 'convolutional':
                    cfg[i-2]['out_size'] = cfg[i-1]['out_width'] * cfg[i-1]['out_height'] * out_ch
                else:
                    raise Exception("???")

        else:
            continue

        layer['in_width'] = in_width
        layer['in_height'] = in_height
        layer['in_framesize'] = in_width * in_height
        layer['in_ch'] = in_ch
        layer['out_width'] = out_width
        layer['out_height'] = out_height
        layer['out_ch'] = out_ch
        layer['weight_size'] = weight_size
        layer['scale_bias_size'] = scale_bias_size
        layer['param_size'] = param_size
        layer['size'] = size
        layer['npu_idx'] = num_layer-1
        layer['out_size'] = out_width * out_height * out_ch

        # Initialize next layer
        in_width = out_width
        in_height = out_height
        in_ch = out_ch

    print("Total param size: %d"%total_param_size)

    ### DRAM Address Management #########################################################
    # TODO: Need to check activation size < 0x01000000(=2^24)
    dram_addr = []
    dram_addr.append(DRAM_BASE_ADDR + total_param_size)             # dram_addr[0]
    dram_addr.append(DRAM_BASE_ADDR + total_param_size + 0x01000000) # dram_addr[1]
    dram_addr.append(DRAM_BASE_ADDR + total_param_size + 0x02000000) # dram_addr[2]
    dram_addr.append(DRAM_BASE_ADDR + total_param_size + 0x03000000) # dram_addr[3]

    # Save DRAM
    save_dram_cnt = 0
    for idx,layer in enumerate(cfg):
        if layer['type'] == 'route':
            route_indices = layer['layers']
            for route_idx in route_indices:
                if cfg[route_idx]['type'] == 'convolutional':
                    cfg[route_idx]['save_dram_addr'] = dram_addr[3] + save_dram_cnt * 0x00200000
                    print("layer_idx:%d, save_dram_cnt:%d, input_layer_idx:%d"%(
                        cfg[idx+1]['npu_idx'],save_dram_cnt,cfg[route_idx]['npu_idx']))
                    save_dram_cnt += 1
                elif cfg[route_idx]['type'] == 'maxpool' or cfg[route_idx]['type'] == 'shortcut':
                    cfg[route_idx-1]['save_dram_addr'] = dram_addr[3] + save_dram_cnt * 0x00200000
                    print("layer_idx:%d, save_dram_cnt:%d, input_layer_idx:%d"%(
                        cfg[idx+1]['npu_idx'],save_dram_cnt,cfg[route_idx-1]['npu_idx']))
                    save_dram_cnt += 1
                else:
                    # if upsample, don't save to dram, because it's right before route
                    continue
        elif idx < len(cfg)-1 and (cfg[idx+1]['type'] == 'yolo' or cfg[idx+1]['type'] == 'region'):
            cfg[idx]['save_dram_addr'] = dram_addr[3] + save_dram_cnt * 0x00200000
            print("layer_idx:%d, save_dram_cnt:%d"%(cfg[idx]['npu_idx'], save_dram_cnt))
            save_dram_cnt += 1
    #####################################################################################

    # NPU config header
    npu_cfg.write("#include \"npu_cfg.h\"\n\n")
    npu_cfg.write("const uint32_t npu_cfg[NUM_OF_LAYERS][NUM_OF_REGS] =\n")
    npu_cfg.write("{\n")

    w_offset_addr = DRAM_BASE_ADDR
    npu_idx = 0
    buff_sel_in, buff_sel_out, buff_sel_sc, buff_sel_sc_prev = 0,0,0,0

    for idx,layer in enumerate(cfg):
        if layer['type'] == 'convolutional':
            in_width = layer['in_width']
            in_height = layer['in_height']
            in_framesize = layer['in_framesize']
            in_ch = layer['in_ch']
            out_ch = layer['out_ch']
            stride = layer['stride']-1  # 0 if stride=1, 1 if stride=2
            if 'depthwise' in layer: # jckim [2023.04.19]
                is_depthwise = layer['depthwise']
            else:
                is_depthwise = 0
            filter_size = layer['size']
            param_size = layer['param_size']

            if layer['type'] == 'convolutional':
                bias_shift = layer['bias_shift']
                act_shift = layer['act_shift']
            else:
                bias_shift = 0
                act_shift = 0

            weight_st_addr = w_offset_addr
            param_st_addr = w_offset_addr + layer['weight_size']
            w_offset_addr += param_size

            next_layer = cfg[idx+1]

            if next_layer['type'] == 'maxpool':
                maxpool_stride = 2-next_layer['stride']
            else:
                maxpool_stride = 0

            # cutpoint
            if npu_idx < CUTPOINT:
                dataflow_en = 0
                is_cutlayer = 0
            elif npu_idx == CUTPOINT:
                dataflow_en = 1
                is_cutlayer = 1
            else:
                dataflow_en = 1
                is_cutlayer = 0

            # is_1st_layer
            if npu_idx == 0:
                is_1st_layer = 1
            else:
                is_1st_layer = 0

            # is_last_layer
            if (next_layer['type'] == 'yolo') or (next_layer['type'] == 'region'):
                is_last_layer = 1
            else:
                is_last_layer = 0

            # is_conv1x1
            if filter_size == 1:
                is_conv1x1 = 1
            else:
                is_conv1x1 = 0

            # maxpool_en
            if next_layer['type'] == 'maxpool':
                maxpool_en = 1
            else:
                maxpool_en = 0

            # avgpool_en
            if next_layer['type'] == 'avgpool':
                avgpool_en = 1
            else:
                avgpool_en = 0

            # upsample_en
            if next_layer['type'] == 'upsample':
                upsample_en = 1
            else:
                upsample_en = 0

            # act_type
            if layer['type'] == 'convolutional':
                if layer['activation'] == 'leaky':
                    act_type = 0
                elif layer['activation'] == 'relu':
                    act_type = 1
                elif layer['activation'] == 'linear':
                    act_type = 2
                else:
                    act_type = 0
            else:
                act_type = 0

            # store_dram_en
            if 'save_dram_addr' in layer:
                store_dram_en = 1
            else:
                store_dram_en = 0

            # eltwise_en, act_elt
            if next_layer['type'] == 'shortcut':
                eltwise_en = 1
                if next_layer['activation'] == 'leaky':
                    act_elt = 0
                elif next_layer['activation'] == 'relu':
                    act_elt = 1
                elif next_layer['activation'] == 'linear':
                    act_elt = 2
                else:
                    act_elt = 0
            else:
                eltwise_en = 0
                act_elt = 0

            # in_act_unsign
            if is_1st_layer:
                in_act_unsign = 1
                # TODO: in_act_unsgin = 0 for EfficientNet 1st layer
            else:
                in_act_unsign = out_act_unsign

            # out_act_unsign
            out_act_unsign = 0
            if eltwise_en:
                if layer['activation'] == 'relu' or next_layer['activation'] == 'relu':
                    out_act_unsign = 1
            else: # jckim [2023.04.19]
                if layer['activation'] == 'relu':
                    out_act_unsign = 1

            # skip_pdma
            if MULTI_CHIP:
                if ('depthwise' in next_layer) and (dataflow_en): # jckim [2023.05.26]
                    skip_pdma = 1
                elif (is_last_layer): # load_dram_en
                    skip_pdma = 1
                elif (not dataflow_en) and (npu_idx != CUTPOINT-1): # cmpark [2023.07.07]
                    skip_pdma = 1
                else:
                    skip_pdma = 0
            else:
                    skip_pdma = 0
            ########################################################################################


            ### Buffer Management(DRAM) ############################################################
            buff_sel_sc_prev = buff_sel_sc
            if not dataflow_en: # row-reuse
                if is_1st_layer:
                    buff_sel_in, buff_sel_out, buff_sel_sc = 0,1,0
                else:
                    buff_sel_in = buff_sel_out
                    for j in range(3):
                        if buff_sel_in != j and buff_sel_sc_prev != j:
                            buff_sel_out = j
                            break
                    if 'to_residual' in layer:
                        buff_sel_sc = buff_sel_out
            elif is_cutlayer:
                if 'to_residual' in cfg[idx-1]:
                    buff_sel_in, buff_sel_out, buff_sel_sc = 2,0,2
                elif 'to_residual' in layer:
                    buff_sel_in, buff_sel_out, buff_sel_sc = 0,2,2
                else:
                    buff_sel_in, buff_sel_out = 0,1
            else: # frame-reuse
                buff_sel_in = buff_sel_out
                if eltwise_en and buff_sel_sc_prev != buff_sel_in:
                    buff_sel_out = buff_sel_sc_prev
                else:
                    for j in range(3):
                        if buff_sel_in != j:
                            buff_sel_out = j
                            break
                if 'to_residual' in layer:
                    if buff_sel_in == 2:
                        buff_sel_out = 0
                    else:
                        buff_sel_out = 2
                    buff_sel_sc = buff_sel_out
            ########################################################################################


            # input_addr, input_addr1, output_addr, output_addr1 ###################################
            save_addr1 = 0
            output_addr1 = 0
            # row-reuse
            if npu_idx < CUTPOINT-1:
                buf_idx = 1
                allocate(layer['param_size'], [npu_idx], buf_idx)
                input_addr = dram_addr[buff_sel_in]

                if eltwise_en:
                   input_addr1 = dram_addr[buff_sel_sc_prev]
                else:
                   input_addr1 = 0

                if 'to_residual' in layer and layer['to_residual'] >= CUTPOINT:
                    output_addr = allocate(layer['out_size'], [layer['to_residual']], buf_idx)
                else:
                    output_addr = dram_addr[buff_sel_out]
            elif npu_idx == CUTPOINT-1:
                buf_idx = 1
                allocate(layer['param_size'], [npu_idx], buf_idx)
                input_addr = dram_addr[buff_sel_in]

                if eltwise_en:
                   input_addr1 = dram_addr[buff_sel_sc_prev]
                else:
                   input_addr1 = 0

                if 1 not in [alloc_data[3] for alloc_data in allocated]:
                    buf_idx = 1
                elif 2 not in [alloc_data[3] for alloc_data in allocated]:
                    buf_idx = 2
                else:
                    buf_idx = 1
                if 'to_residual' in layer:
                    output_addr = allocate(layer['out_size'], [npu_idx+1,layer['to_residual']], buf_idx)
                else:
                    output_addr = allocate(layer['out_size'], [npu_idx+1], buf_idx)
            # frame-reuse
            else:
                if eltwise_en:
                    input_addr = search(npu_idx,2)[1]
                    input_addr1 = search(npu_idx,2)[0]
                else:
                    if 'route' in layer:
                        if search(npu_idx,1) is None:
                            # input data saved at DRAM
                            input_addr = cfg[layer['route'][0]]['save_dram_addr']
                        else:
                            # input data saved at buffer
                            input_addr = search(npu_idx,1)[0]
                    else:
                        input_addr = search(npu_idx,1)[0]
                    input_addr1 = 0

                # Save output in ping-pong buffer
#                buf_idx = 1
#                buf_idx = (npu_idx-CUTPOINT) % 2 + 1
                if 1 not in [alloc_data[3] for alloc_data in allocated]:
                    buf_idx = 1
                elif 2 not in [alloc_data[3] for alloc_data in allocated]:
                    buf_idx = 2
                else:
                    buf_idx = 1

                if is_last_layer:
                    output_addr = layer['save_dram_addr']
                elif 'to_residual' in layer:
                    output_addr = allocate(layer['out_size'], [npu_idx+1,layer['to_residual']], buf_idx)
                else:
                    output_addr = allocate(layer['out_size'], [npu_idx+1], buf_idx)

                if 'save_dram_addr' in layer and not is_last_layer:
                    # Save OFM to both buffer & DRAM
                    save_addr1 = 1
                    output_addr1 = layer['save_dram_addr'] # DRAM addr
            #total_size = 0
            #for addr,size,layers in allocated:
            #   total_size += size
            #print(npu_idx,":",total_size)
            print(npu_idx,":",allocated)
            free(npu_idx)
            ########################################################################################


            # load_dram_en, concat_en, offset_wsize, offset_waddr ##################################
            if 'route' in layer:
                route_indices = layer['route']
                if len(route_indices) == 1:
                    load_dram_en = 1
                    concat_en = 0
                    offset_wsize = 0
                    offset_waddr = 0
                elif len(route_indices) == 2:
                    load_dram_en = 0
                    concat_en = 1
                    layer0 = cfg[route_indices[0]] # late layer comes first, already saved in buffer
                    layer1 = cfg[route_indices[1]] # early layer goes behind, load from DRAM
                    offset_waddr = layer0['out_width']*layer0['out_height']*layer0['out_ch']//PF
                    offset_wsize = layer1['out_width']*layer1['out_height']*layer1['out_ch']//PF
                    if cfg[route_indices[1]]['type'] == 'convolutional':
                        input_addr1 = cfg[route_indices[1]]['save_dram_addr']
                    elif cfg[route_indices[1]-1]['type'] == 'convolutional':
                        input_addr1 = cfg[route_indices[1]-1]['save_dram_addr']
                    else:
                        raise Exception("???")
                else:
                    raise NotImplementedError

            else:
                load_dram_en, concat_en, offset_wsize, offset_waddr = 0,0,0,0
            ########################################################################################


            # Multi chip ###########################################################################
            if MULTI_CHIP:
                if maxpool_en or upsample_en:
                   out_width = next_layer['out_width']
                   out_height = next_layer['out_height']
                else:
                   out_width = layer['out_width']
                   out_height = layer['out_height']

                # skip_last_row, skip_1st_row
                if CHIP_NUM == 0:
                    skip_last_row,skip_1st_row = 1,0
                elif CHIP_NUM == 1:
                    skip_last_row,skip_1st_row = 0,1
                else:
                    raise NotImplementedError

                if not dataflow_en: # row-reuse
                    if CHIP_NUM == 1:
                        # input_addr
                        if npu_idx == 0: # 1st layer optimzation
                            input_addr = input_addr + in_width * (in_height//2 - 1) * 4
                        else:
                            input_addr = input_addr + in_width * (in_height//2 - 1) * in_ch

                        # input_addr1
                        if eltwise_en:
                           input_addr1 = input_addr1 + out_width * (out_height//2) * out_ch

                        # output_addr
                        output_addr = output_addr + out_width * (out_height//2) * out_ch

                    # in_height, in_framesize
                    in_height = in_height//2 + 1
                    in_framesize = in_height * in_width

                    # CDMA: load weight from DRAM to NPU_MEM
                    cdma_src_addr = weight_st_addr
                    cdma_dst_addr = 0x60000000
                    cdma_num_trans = param_size //PF

                else: # frame-reuse
                    # out_ch_chip0, out_ch_chip1
                    out_ch_chip0 = ((out_ch//2+PF-1)//PF)*PF
                    out_ch_chip1 = out_ch - out_ch_chip0
                    if CHIP_NUM == 0:
                        out_ch = out_ch_chip0
                    else: # CHIP_NUM = 1
                        out_ch = out_ch_chip1

                    # cdma_src_addr, cdma_dst_addr, cdma_num_trans
                    if load_dram_en:
                        cdma_src_addr = 0
                        cdma_dst_addr = 0
                        cdma_num_trans = 0
                    elif is_cutlayer:
                        if CHIP_NUM == 0:
                            cdma_dst_addr = input_addr + (in_height*in_width*in_ch) //2
                            cdma_src_addr = (cdma_dst_addr & 0x0FFFFFFF) | 0x50000000 # addr translation
                            cdma_num_trans = (in_height*in_width*in_ch//2) //PF
                        else: # CHIP_NUM == 1
                            cdma_dst_addr = input_addr
                            cdma_src_addr = (cdma_dst_addr & 0x0FFFFFFF) | 0x50000000 # addr translation
                            cdma_num_trans = (in_height*in_width*in_ch//2) //PF
                    else:
                        if CHIP_NUM == 0:
                            # out_ch_chip0_prev : out_ch of previous layer in chip0
                            cdma_dst_addr = input_addr + in_height*in_width*out_ch_chip0_prev
                            cdma_src_addr = (cdma_dst_addr & 0x0FFFFFFF) | 0x50000000 # addr translation
                            cdma_num_trans = (in_height*in_width*out_ch_chip1_prev) //PF
                        else: # CHIP_NUM == 1
                            cdma_dst_addr = input_addr
                            cdma_src_addr = (cdma_dst_addr & 0x0FFFFFFF) | 0x50000000 # addr translation
                            cdma_num_trans = (in_height*in_width*out_ch_chip0_prev) //PF

                    # weight_st_addr, param_st_addr
                    if CHIP_NUM == 1:
                        if is_depthwise: # jckim [2023.04.22]
                            weight_st_addr = weight_st_addr + filter_size*filter_size*out_ch_chip0
                            param_st_addr = param_st_addr + 4 * out_ch_chip0
                        else:
                            weight_st_addr = weight_st_addr + filter_size*filter_size*in_ch*out_ch_chip0
                            param_st_addr = param_st_addr + 4 * out_ch_chip0

                    # input_addr
                    if CHIP_NUM == 1:
                        if is_depthwise: # jckim [2023.04.25]
                           input_addr = input_addr + in_width * in_height  * (in_ch//2)

                    # input_addr1
                    if CHIP_NUM == 1:
                        if eltwise_en:
                           input_addr1 = input_addr1 + out_width * out_height * out_ch_chip0

                    # output_addr
                    if CHIP_NUM == 1:
                        output_addr = output_addr + out_width * out_height * out_ch_chip0

                    # output_addr1
                    if CHIP_NUM == 1:
                        if 'save_dram_addr' in layer and not is_last_layer:
                            output_addr1 = output_addr1 + out_width * out_height * out_ch_chip0

                    out_ch_chip0_prev = out_ch_chip0
                    out_ch_chip1_prev = out_ch_chip1

            else: # single-chip
                skip_last_row = 0
                skip_1st_row = 0

                if not dataflow_en: # row-reuse
                    cdma_src_addr = weight_st_addr
                    cdma_dst_addr = 0x60000000
                    cdma_num_trans = param_size //PF
                else: # frame-reuse
                    cdma_src_addr = 0
                    cdma_dst_addr = 0
                    cdma_num_trans = 0
            ########################################################################################


            # Register #############################################################################
            regs = []
            # reg[0]
            regs.append( skip_last_row<<31 | skip_1st_row<<30 | skip_pdma<<29 | in_height<<12 | in_width )
            # reg[1]
            regs.append( in_framesize )
            # reg[2]
            regs.append( out_ch<<18 | in_ch<<5 | maxpool_stride<<4 |
                         stride<<3 | filter_size )
            # reg[3]
            regs.append( cdma_num_trans )
            # reg[4]
            regs.append( weight_st_addr )
            # reg[5]
            regs.append( param_st_addr )
            # reg[6]
            regs.append( in_act_unsign<<31 | bias_shift<<26 | act_shift<<23 |
                         act_elt<<21 | buff_sel_sc_prev<<19 | act_type<<17 |
                         eltwise_en<<16 | is_last_layer<<15 | is_1st_layer<<14 |
                         upsample_en<<13 | avgpool_en<<12 | maxpool_en<<11 |
                         is_conv1x1<<10 | buff_sel_out<<8 | buff_sel_in<<6 |
                         is_depthwise<<5 | concat_en<<4 | is_cutlayer<<3 |
                         load_dram_en<<2 | save_addr1<<1 | dataflow_en )
            # reg[7]
            regs.append( input_addr )
            # reg[8]
            regs.append( input_addr1 )
            # reg[9]
            regs.append( output_addr )
            # reg[10]
            regs.append( offset_wsize << 16 | offset_waddr )
            # reg[11]
            regs.append( output_addr1 )
            # reg[12]
            regs.append( cdma_src_addr )
            # reg[13]
            regs.append( cdma_dst_addr )

            npu_cfg.write("    { // layer%d: %dx%dconv ("%(npu_idx,filter_size,filter_size))
            if stride:
                npu_cfg.write("stride,")
            if save_addr1:
                npu_cfg.write("save_addr1,")
            if load_dram_en:
                npu_cfg.write("load_dram,")
            if is_cutlayer:
                npu_cfg.write("cutlayer,")
            if concat_en:
                npu_cfg.write("concat,")
            if is_depthwise:
                npu_cfg.write("depthwise,")
            if maxpool_en:
                npu_cfg.write("maxpool,")
            if upsample_en:
                npu_cfg.write("upsample,")
            if eltwise_en:
                npu_cfg.write("eltwise,")
            if is_last_layer:
                npu_cfg.write("last_layer,")
            if skip_pdma:
                npu_cfg.write("skip_pdma,")
            npu_cfg.write(")\n")
            for reg_idx,reg in enumerate(regs):
                if reg_idx < len(regs)-1:
                    if (reg_idx == 0):
                        npu_cfg.write("        0x%08x, // h,w=(%d,%d)\n"%(reg,in_height,in_width))
                    elif (reg_idx == 1):
                        npu_cfg.write("        0x%08x, // frame_size=%d\n"%(reg,in_framesize))
                    elif (reg_idx == 2):
                        npu_cfg.write("        0x%08x, // in_ch,out_ch=(%d,%d)\n"%(reg,in_ch,out_ch))
                    elif (reg_idx == 3):
                        npu_cfg.write("        0x%08x, // cdma_num_trans\n"%(reg))
                    elif (reg_idx == 4):
                        npu_cfg.write("        0x%08x, // weight_st_addr\n"%(reg))
                    elif (reg_idx == 5):
                        npu_cfg.write("        0x%08x, // param_st_addr\n"%(reg))
                    elif (reg_idx == 6):
                        npu_cfg.write("        0x%08x,\n"%(reg))
                    elif (reg_idx == 7):
                        npu_cfg.write("        0x%08x, // input_addr\n"%(reg))
                    elif (reg_idx == 8):
                        npu_cfg.write("        0x%08x, // input_addr1\n"%(reg))
                    elif (reg_idx == 9):
                        npu_cfg.write("        0x%08x, // output_addr\n"%(reg))
                    elif (reg_idx == 10):
                        npu_cfg.write("        0x%08x, // offset_w\n"%(reg))
                    elif (reg_idx == 11):
                        npu_cfg.write("        0x%08x, // output_addr1\n"%(reg))
                    elif (reg_idx == 12):
                        npu_cfg.write("        0x%08x, // cdma_src_addr\n"%(reg))
                    else:
                        npu_cfg.write("        0x%08x,\n"%(reg))
                else: # reg_idx == 13
                    npu_cfg.write("        0x%08x  // cdma_dst_addr\n"%(reg))
            if npu_idx < num_layer-1:
                npu_cfg.write("    },\n")
            else:
                npu_cfg.write("    }\n")
            ########################################################################################

#            if not dataflow_en:
#               dataflow = "row"
#            else:
#               dataflow = "frame"
#            print("{} {} {} {} {} {} {} {} {} {} {}".format(npu_idx, dataflow, is_depthwise, maxpool_stride, eltwise_en, filter_size, in_ch, out_ch, stride+1, in_height, in_width))
            npu_idx += 1
    ## End of for loop

    npu_cfg.write("};\n")
    print("Buffer1 requirement:", max_addr1,"Bytes")
    print("Buffer2 requirement:", max_addr2,"Bytes")
    print("Buffer3 requirement:", max_addr3,"Bytes")
