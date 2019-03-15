import numpy as np
logx = np.array([-4.0, -3.994666666666667, -3.989333333333333, -3.984, -3.978666666666667, -3.973333333333333, -3.968, -3.962666666666667, -3.957333333333333, -3.952, -3.9466666666666668, -3.941333333333333, -3.936, -3.9306666666666668, -3.9253333333333336, -3.92, -3.9146666666666667, -3.9093333333333335, -3.904, -3.8986666666666667, -3.8933333333333335, -3.888, -3.8826666666666667, -3.8773333333333335, -3.872, -3.8666666666666667, -3.8613333333333335, -3.856, -3.8506666666666667, -3.8453333333333335, -3.84, -3.8346666666666667, -3.8293333333333335, -3.824, -3.8186666666666667, -3.8133333333333335, -3.808, -3.8026666666666666, -3.7973333333333334, -3.792, -3.7866666666666666, -3.7813333333333334, -3.776, -3.7706666666666666, -3.7653333333333334, -3.76, -3.7546666666666666, -3.7493333333333334, -3.7439999999999998, -3.7386666666666666, -3.7333333333333334, -3.728, -3.7226666666666666, -3.7173333333333334, -3.712, -3.7066666666666666, -3.7013333333333334, -3.696, -3.6906666666666665, -3.6853333333333333, -3.68, -3.6746666666666665, -3.6693333333333333, -3.664, -3.6586666666666665, -3.6533333333333333, -3.648, -3.6426666666666665, -3.6373333333333333, -3.632, -3.626666666666667, -3.6213333333333333, -3.616, -3.610666666666667, -3.6053333333333333, -3.6, -3.594666666666667, -3.5893333333333333, -3.584, -3.578666666666667, -3.5733333333333333, -3.568, -3.562666666666667, -3.5573333333333332, -3.552, -3.546666666666667, -3.541333333333333, -3.536, -3.530666666666667, -3.525333333333333, -3.52, -3.514666666666667, -3.509333333333333, -3.504, -3.498666666666667, -3.493333333333333, -3.488, -3.482666666666667, -3.477333333333333, -3.472, -3.466666666666667, -3.461333333333333, -3.456, -3.4506666666666668, -3.445333333333333, -3.44, -3.4346666666666668, -3.429333333333333, -3.424, -3.4186666666666667, -3.413333333333333, -3.408, -3.4026666666666667, -3.397333333333333, -3.392, -3.3866666666666667, -3.381333333333333, -3.376, -3.3706666666666667, -3.3653333333333335, -3.36, -3.3546666666666667, -3.3493333333333335, -3.344, -3.3386666666666667, -3.3333333333333335, -3.3280000000000003, -3.3226666666666667, -3.3173333333333335, -3.3120000000000003, -3.3066666666666666, -3.3013333333333335, -3.2960000000000003, -3.2906666666666666, -3.2853333333333334, -3.2800000000000002, -3.2746666666666666, -3.2693333333333334, -3.2640000000000002, -3.2586666666666666, -3.2533333333333334, -3.248, -3.2426666666666666, -3.2373333333333334, -3.232, -3.2266666666666666, -3.2213333333333334, -3.216, -3.2106666666666666, -3.2053333333333334, -3.2, -3.1946666666666665, -3.1893333333333334, -3.184, -3.1786666666666665, -3.1733333333333333, -3.168, -3.1626666666666665, -3.1573333333333333, -3.152, -3.1466666666666665, -3.1413333333333333, -3.136, -3.1306666666666665, -3.1253333333333333, -3.12, -3.1146666666666665, -3.1093333333333333, -3.104, -3.0986666666666665, -3.0933333333333333, -3.088, -3.0826666666666664, -3.0773333333333333, -3.072, -3.0666666666666664, -3.0613333333333332, -3.056, -3.050666666666667, -3.0453333333333332, -3.04, -3.034666666666667, -3.029333333333333, -3.024, -3.018666666666667, -3.0133333333333336, -3.008, -3.002666666666667, -2.9973333333333336, -2.992, -2.986666666666667, -2.9813333333333336, -2.976, -2.970666666666667, -2.9653333333333336, -2.96, -2.9546666666666668, -2.9493333333333336, -2.944, -2.9386666666666668, -2.9333333333333336, -2.928, -2.9226666666666667, -2.9173333333333336, -2.912, -2.9066666666666667, -2.9013333333333335, -2.896, -2.8906666666666667, -2.8853333333333335, -2.88, -2.8746666666666667, -2.8693333333333335, -2.864, -2.8586666666666667, -2.8533333333333335, -2.848, -2.8426666666666667, -2.8373333333333335, -2.832, -2.8266666666666667, -2.8213333333333335, -2.816, -2.8106666666666666, -2.8053333333333335, -2.8, -2.7946666666666666, -2.7893333333333334, -2.784, -2.7786666666666666, -2.7733333333333334, -2.768, -2.7626666666666666, -2.7573333333333334, -2.752, -2.746666666666667, -2.7413333333333334, -2.7359999999999998, -2.730666666666667, -2.7253333333333334, -2.7199999999999998, -2.714666666666667, -2.7093333333333334, -2.7039999999999997, -2.698666666666667, -2.6933333333333334, -2.6879999999999997, -2.682666666666667, -2.6773333333333333, -2.672, -2.666666666666667, -2.6613333333333333, -2.656, -2.650666666666667, -2.6453333333333333, -2.64, -2.634666666666667, -2.6293333333333333, -2.624, -2.618666666666667, -2.6133333333333333, -2.608, -2.602666666666667, -2.5973333333333333, -2.592, -2.586666666666667, -2.5813333333333333, -2.576, -2.570666666666667, -2.5653333333333332, -2.56, -2.554666666666667, -2.5493333333333332, -2.544, -2.538666666666667, -2.533333333333333, -2.528, -2.522666666666667, -2.517333333333333, -2.512, -2.506666666666667, -2.501333333333333, -2.496, -2.490666666666667, -2.485333333333333, -2.48, -2.474666666666667, -2.469333333333333, -2.464, -2.458666666666667, -2.453333333333333, -2.448, -2.4426666666666668, -2.437333333333333, -2.4320000000000004, -2.4266666666666667, -2.421333333333333, -2.4160000000000004, -2.4106666666666667, -2.405333333333333, -2.4000000000000004, -2.3946666666666667, -2.389333333333333, -2.3840000000000003, -2.3786666666666667, -2.373333333333333, -2.3680000000000003, -2.3626666666666667, -2.357333333333333, -2.3520000000000003, -2.3466666666666667, -2.3413333333333335, -2.3360000000000003, -2.3306666666666667, -2.3253333333333335, -2.3200000000000003, -2.3146666666666667, -2.3093333333333335, -2.3040000000000003, -2.2986666666666666, -2.2933333333333334, -2.2880000000000003, -2.2826666666666666, -2.2773333333333334, -2.2720000000000002, -2.2666666666666666, -2.2613333333333334, -2.2560000000000002, -2.2506666666666666, -2.2453333333333334, -2.24, -2.2346666666666666, -2.2293333333333334, -2.224, -2.2186666666666666, -2.2133333333333334, -2.208, -2.2026666666666666, -2.1973333333333334, -2.192, -2.1866666666666665, -2.1813333333333333, -2.176, -2.1706666666666665, -2.1653333333333333, -2.16, -2.1546666666666665, -2.1493333333333333, -2.144, -2.1386666666666665, -2.1333333333333333, -2.128, -2.1226666666666665, -2.1173333333333333, -2.112, -2.1066666666666665, -2.1013333333333337, -2.096, -2.0906666666666665, -2.0853333333333337, -2.08, -2.0746666666666664, -2.0693333333333337, -2.064, -2.0586666666666664, -2.0533333333333337, -2.048, -2.0426666666666664, -2.0373333333333337, -2.032, -2.026666666666667, -2.0213333333333336, -2.016, -2.010666666666667, -2.0053333333333336, -2.0, -1.9946666666666668, -1.9893333333333336, -1.984, -1.9786666666666668, -1.9733333333333336, -1.968, -1.9626666666666668, -1.9573333333333336, -1.952, -1.9466666666666668, -1.9413333333333336, -1.936, -1.9306666666666668, -1.9253333333333336, -1.92, -1.9146666666666667, -1.9093333333333335, -1.904, -1.8986666666666667, -1.8933333333333335, -1.888, -1.8826666666666667, -1.8773333333333335, -1.8719999999999999, -1.8666666666666667, -1.8613333333333335, -1.8559999999999999, -1.8506666666666667, -1.8453333333333335, -1.8399999999999999, -1.8346666666666667, -1.8293333333333335, -1.8240000000000003, -1.8186666666666667, -1.8133333333333335, -1.8080000000000003, -1.8026666666666666, -1.7973333333333334, -1.7920000000000003, -1.7866666666666666, -1.7813333333333334, -1.7760000000000002, -1.7706666666666666, -1.7653333333333334, -1.7600000000000002, -1.7546666666666666, -1.7493333333333334, -1.7440000000000002, -1.7386666666666666, -1.7333333333333334, -1.7280000000000002, -1.7226666666666666, -1.7173333333333334, -1.7120000000000002, -1.7066666666666666, -1.7013333333333334, -1.6960000000000002, -1.6906666666666665, -1.6853333333333333, -1.6800000000000002, -1.6746666666666665, -1.6693333333333333, -1.6640000000000001, -1.658666666666667, -1.6533333333333333, -1.6480000000000001, -1.642666666666667, -1.6373333333333333, -1.6320000000000001, -1.626666666666667, -1.6213333333333333, -1.616, -1.610666666666667, -1.6053333333333333, -1.6, -1.594666666666667, -1.5893333333333333, -1.584, -1.5786666666666669, -1.5733333333333333, -1.568, -1.5626666666666669, -1.5573333333333332, -1.552, -1.5466666666666669, -1.5413333333333332, -1.536, -1.5306666666666668, -1.5253333333333332, -1.52, -1.5146666666666668, -1.5093333333333336, -1.504, -1.4986666666666668, -1.4933333333333336, -1.488, -1.4826666666666668, -1.4773333333333336, -1.472, -1.4666666666666668, -1.4613333333333336, -1.456, -1.4506666666666668, -1.4453333333333336, -1.44, -1.4346666666666668, -1.4293333333333336, -1.424, -1.4186666666666667, -1.4133333333333336, -1.408, -1.4026666666666667, -1.3973333333333335, -1.392, -1.3866666666666667, -1.3813333333333335, -1.376, -1.3706666666666667, -1.3653333333333335, -1.3599999999999999, -1.3546666666666667, -1.3493333333333335, -1.3440000000000003, -1.3386666666666667, -1.3333333333333335, -1.3280000000000003, -1.3226666666666667, -1.3173333333333335, -1.3120000000000003, -1.3066666666666666, -1.3013333333333335, -1.2960000000000003, -1.2906666666666666, -1.2853333333333334, -1.2800000000000002, -1.2746666666666666, -1.2693333333333334, -1.2640000000000002, -1.2586666666666666, -1.2533333333333334, -1.2480000000000002, -1.2426666666666666, -1.2373333333333334, -1.2320000000000002, -1.2266666666666666, -1.2213333333333334, -1.2160000000000002, -1.2106666666666666, -1.2053333333333334, -1.2000000000000002, -1.1946666666666665, -1.1893333333333334, -1.1840000000000002, -1.178666666666667, -1.1733333333333333, -1.1680000000000001, -1.162666666666667, -1.1573333333333333, -1.1520000000000001, -1.146666666666667, -1.1413333333333333, -1.1360000000000001, -1.130666666666667, -1.1253333333333333, -1.12, -1.114666666666667, -1.1093333333333333, -1.104, -1.098666666666667, -1.0933333333333333, -1.088, -1.0826666666666669, -1.0773333333333333, -1.072, -1.0666666666666669, -1.0613333333333332, -1.056, -1.0506666666666669, -1.0453333333333332, -1.04, -1.0346666666666668, -1.0293333333333332, -1.024, -1.0186666666666668, -1.0133333333333336, -1.008, -1.0026666666666668, -0.9973333333333336, -0.992, -0.9866666666666668, -0.9813333333333336, -0.976, -0.9706666666666668, -0.9653333333333336, -0.96, -0.9546666666666668, -0.9493333333333336, -0.944, -0.9386666666666668, -0.9333333333333336, -0.9279999999999999, -0.9226666666666667, -0.9173333333333336, -0.9119999999999999, -0.9066666666666667, -0.9013333333333335, -0.8959999999999999, -0.8906666666666667, -0.8853333333333335, -0.8799999999999999, -0.8746666666666667, -0.8693333333333335, -0.8640000000000003, -0.8586666666666667, -0.8533333333333335, -0.8480000000000003, -0.8426666666666667, -0.8373333333333335, -0.8320000000000003, -0.8266666666666667, -0.8213333333333335, -0.8160000000000003, -0.8106666666666666, -0.8053333333333335, -0.8000000000000003, -0.7946666666666666, -0.7893333333333334, -0.7840000000000003, -0.7786666666666666, -0.7733333333333334, -0.7680000000000002, -0.7626666666666666, -0.7573333333333334, -0.7520000000000002, -0.7466666666666666, -0.7413333333333334, -0.7360000000000002, -0.7306666666666666, -0.7253333333333334, -0.7200000000000002, -0.7146666666666666, -0.7093333333333334, -0.7040000000000002, -0.698666666666667, -0.6933333333333334, -0.6880000000000002, -0.682666666666667, -0.6773333333333333, -0.6720000000000002, -0.666666666666667, -0.6613333333333333, -0.6560000000000001, -0.650666666666667, -0.6453333333333333, -0.6400000000000001, -0.6346666666666669, -0.6293333333333333, -0.6240000000000001, -0.6186666666666669, -0.6133333333333333, -0.6080000000000001, -0.6026666666666669, -0.5973333333333333, -0.5920000000000001, -0.5866666666666669, -0.5813333333333333, -0.5760000000000001, -0.5706666666666669, -0.5653333333333332, -0.56, -0.5546666666666669, -0.5493333333333332, -0.544, -0.5386666666666668, -0.5333333333333337, -0.528, -0.5226666666666668, -0.5173333333333336, -0.512, -0.5066666666666668, -0.5013333333333336, -0.496, -0.4906666666666668, -0.4853333333333336, -0.48, -0.4746666666666668, -0.4693333333333336, -0.46399999999999997, -0.4586666666666668, -0.4533333333333336, -0.44799999999999995, -0.44266666666666676, -0.4373333333333336, -0.43199999999999994, -0.42666666666666675, -0.42133333333333356, -0.4159999999999999, -0.41066666666666674, -0.40533333333333355, -0.3999999999999999, -0.3946666666666667, -0.38933333333333353, -0.38400000000000034, -0.3786666666666667, -0.3733333333333335, -0.3680000000000003, -0.3626666666666667, -0.3573333333333335, -0.3520000000000003, -0.3466666666666667, -0.3413333333333335, -0.3360000000000003, -0.33066666666666666, -0.3253333333333335, -0.3200000000000003, -0.31466666666666665, -0.30933333333333346, -0.30400000000000027, -0.29866666666666664, -0.29333333333333345, -0.28800000000000026, -0.2826666666666666, -0.27733333333333343, -0.27200000000000024, -0.2666666666666666, -0.2613333333333334, -0.2560000000000002, -0.2506666666666666, -0.2453333333333334, -0.2400000000000002, -0.23466666666666658, -0.2293333333333334, -0.2240000000000002, -0.218666666666667, -0.21333333333333337, -0.20800000000000018, -0.202666666666667, -0.19733333333333336, -0.19200000000000017, -0.18666666666666698, -0.18133333333333335, -0.17600000000000016, -0.17066666666666697, -0.16533333333333333, -0.16000000000000014, -0.15466666666666695, -0.14933333333333332, -0.14400000000000013, -0.13866666666666694, -0.1333333333333333, -0.1280000000000001, -0.12266666666666692, -0.11733333333333329, -0.1120000000000001, -0.10666666666666691, -0.10133333333333328, -0.09600000000000009, -0.0906666666666669, -0.08533333333333326, -0.08000000000000007, -0.07466666666666688, -0.06933333333333325, -0.06400000000000006, -0.05866666666666687, -0.05333333333333368, -0.04800000000000004, -0.04266666666666685, -0.03733333333333366, -0.03200000000000003, -0.02666666666666684, -0.02133333333333365, -0.016000000000000014, -0.010666666666666824, -0.005333333333333634, 0.0, 0.005333333333333634, 0.01066666666666638, 0.016000000000000014, 0.02133333333333276, 0.026666666666666394, 0.03200000000000003, 0.037333333333332774, 0.04266666666666641, 0.04800000000000004, 0.05333333333333279, 0.05866666666666642, 0.06400000000000006, 0.0693333333333328, 0.07466666666666644, 0.08000000000000007, 0.08533333333333282, 0.09066666666666645, 0.09600000000000009, 0.10133333333333283, 0.10666666666666647, 0.1120000000000001, 0.11733333333333285, 0.12266666666666648, 0.1280000000000001, 0.13333333333333286, 0.1386666666666665, 0.14400000000000013, 0.14933333333333287, 0.1546666666666665, 0.16000000000000014, 0.1653333333333329, 0.17066666666666652, 0.17600000000000016, 0.1813333333333329, 0.18666666666666654, 0.19200000000000017, 0.19733333333333292, 0.20266666666666655, 0.20800000000000018, 0.21333333333333293, 0.21866666666666656, 0.2240000000000002, 0.22933333333333294, 0.23466666666666658, 0.2400000000000002, 0.24533333333333296, 0.2506666666666666, 0.2560000000000002, 0.261333333333333, 0.2666666666666666, 0.27200000000000024, 0.277333333333333, 0.2826666666666666, 0.28800000000000026, 0.293333333333333, 0.29866666666666664, 0.30400000000000027, 0.309333333333333, 0.31466666666666665, 0.3200000000000003, 0.32533333333333303, 0.33066666666666666, 0.3360000000000003, 0.34133333333333304, 0.3466666666666667, 0.3519999999999994, 0.35733333333333306, 0.3626666666666667, 0.36799999999999944, 0.3733333333333331, 0.3786666666666667, 0.38399999999999945, 0.3893333333333331, 0.3946666666666667, 0.39999999999999947, 0.4053333333333331, 0.41066666666666674, 0.4159999999999995, 0.4213333333333331, 0.42666666666666675, 0.4319999999999995, 0.43733333333333313, 0.44266666666666676, 0.4479999999999995, 0.45333333333333314, 0.4586666666666668, 0.4639999999999995, 0.46933333333333316, 0.4746666666666668, 0.47999999999999954, 0.4853333333333332, 0.4906666666666668, 0.49599999999999955, 0.5013333333333332, 0.5066666666666668, 0.5119999999999996, 0.5173333333333332, 0.5226666666666668, 0.5279999999999996, 0.5333333333333332, 0.5386666666666668, 0.5439999999999996, 0.5493333333333332, 0.5546666666666669, 0.5599999999999996, 0.5653333333333332, 0.5706666666666669, 0.5759999999999996, 0.5813333333333333, 0.5866666666666669, 0.5919999999999996, 0.5973333333333333, 0.6026666666666669, 0.6079999999999997, 0.6133333333333333, 0.6186666666666669, 0.6239999999999997, 0.6293333333333333, 0.6346666666666669, 0.6399999999999997, 0.6453333333333333, 0.650666666666667, 0.6559999999999997, 0.6613333333333333, 0.6666666666666661, 0.6719999999999997, 0.6773333333333333, 0.6826666666666661, 0.6879999999999997, 0.6933333333333334, 0.6986666666666661, 0.7039999999999997, 0.7093333333333334, 0.7146666666666661, 0.7199999999999998, 0.7253333333333334, 0.7306666666666661, 0.7359999999999998, 0.7413333333333334, 0.7466666666666661, 0.7519999999999998, 0.7573333333333334, 0.7626666666666662, 0.7679999999999998, 0.7733333333333334, 0.7786666666666662, 0.7839999999999998, 0.7893333333333334, 0.7946666666666662, 0.7999999999999998, 0.8053333333333335, 0.8106666666666662, 0.8159999999999998, 0.8213333333333335, 0.8266666666666662, 0.8319999999999999, 0.8373333333333335, 0.8426666666666662, 0.8479999999999999, 0.8533333333333335, 0.8586666666666662, 0.8639999999999999, 0.8693333333333335, 0.8746666666666663, 0.8799999999999999, 0.8853333333333335, 0.8906666666666663, 0.8959999999999999, 0.9013333333333335, 0.9066666666666663, 0.9119999999999999, 0.9173333333333336, 0.9226666666666663, 0.9279999999999999, 0.9333333333333336, 0.9386666666666663, 0.944, 0.9493333333333336, 0.9546666666666663, 0.96, 0.9653333333333336, 0.9706666666666663, 0.976, 0.9813333333333327, 0.9866666666666664, 0.992, 0.9973333333333327, 1.0026666666666664, 1.008, 1.0133333333333328, 1.0186666666666664, 1.024, 1.0293333333333328, 1.0346666666666664, 1.04, 1.0453333333333328, 1.0506666666666664, 1.056, 1.0613333333333328, 1.0666666666666664, 1.072, 1.0773333333333328, 1.0826666666666664, 1.088, 1.0933333333333328, 1.0986666666666665, 1.104, 1.1093333333333328, 1.1146666666666665, 1.12, 1.1253333333333329, 1.1306666666666665, 1.1360000000000001, 1.1413333333333329, 1.1466666666666665, 1.1520000000000001, 1.1573333333333329, 1.1626666666666665, 1.1680000000000001, 1.173333333333333, 1.1786666666666665, 1.1840000000000002, 1.189333333333333, 1.1946666666666665, 1.2000000000000002, 1.205333333333333, 1.2106666666666666, 1.2160000000000002, 1.221333333333333, 1.2266666666666666, 1.2320000000000002, 1.237333333333333, 1.2426666666666666, 1.2480000000000002, 1.253333333333333, 1.2586666666666666, 1.2640000000000002, 1.269333333333333, 1.2746666666666666, 1.2800000000000002, 1.285333333333333, 1.2906666666666666, 1.2960000000000003, 1.301333333333333, 1.3066666666666666, 1.3119999999999994, 1.317333333333333, 1.3226666666666667, 1.3279999999999994, 1.333333333333333, 1.3386666666666667, 1.3439999999999994, 1.349333333333333, 1.3546666666666667, 1.3599999999999994, 1.365333333333333, 1.3706666666666667, 1.3759999999999994, 1.381333333333333, 1.3866666666666667, 1.3919999999999995, 1.397333333333333, 1.4026666666666667, 1.4079999999999995, 1.413333333333333, 1.4186666666666667, 1.4239999999999995, 1.4293333333333331, 1.4346666666666668, 1.4399999999999995, 1.4453333333333331, 1.4506666666666668, 1.4559999999999995, 1.4613333333333332, 1.4666666666666668, 1.4719999999999995, 1.4773333333333332, 1.4826666666666668, 1.4879999999999995, 1.4933333333333332, 1.4986666666666668, 1.5039999999999996, 1.5093333333333332, 1.5146666666666668, 1.5199999999999996, 1.5253333333333332, 1.5306666666666668, 1.5359999999999996, 1.5413333333333332, 1.5466666666666669, 1.5519999999999996, 1.5573333333333332, 1.5626666666666669, 1.5679999999999996, 1.5733333333333333, 1.5786666666666669, 1.5839999999999996, 1.5893333333333333, 1.594666666666667, 1.5999999999999996, 1.6053333333333333, 1.610666666666667, 1.6159999999999997, 1.6213333333333333, 1.626666666666666, 1.6319999999999997, 1.6373333333333333, 1.642666666666666, 1.6479999999999997, 1.6533333333333333, 1.658666666666666, 1.6639999999999997, 1.6693333333333333, 1.674666666666666, 1.6799999999999997, 1.6853333333333333, 1.690666666666666, 1.6959999999999997, 1.7013333333333334, 1.706666666666666, 1.7119999999999997, 1.7173333333333334, 1.7226666666666661, 1.7279999999999998, 1.7333333333333334, 1.7386666666666661, 1.7439999999999998, 1.7493333333333334, 1.7546666666666662, 1.7599999999999998, 1.7653333333333334, 1.7706666666666662, 1.7759999999999998, 1.7813333333333334, 1.7866666666666662, 1.7919999999999998, 1.7973333333333334, 1.8026666666666662, 1.8079999999999998, 1.8133333333333335, 1.8186666666666662, 1.8239999999999998, 1.8293333333333335, 1.8346666666666662, 1.8399999999999999, 1.8453333333333335, 1.8506666666666662, 1.8559999999999999, 1.8613333333333335, 1.8666666666666663, 1.8719999999999999, 1.8773333333333335, 1.8826666666666663, 1.888, 1.8933333333333335, 1.8986666666666663, 1.904, 1.9093333333333335, 1.9146666666666663, 1.92, 1.9253333333333336, 1.9306666666666663, 1.936, 1.9413333333333336, 1.9466666666666663, 1.952, 1.9573333333333327, 1.9626666666666663, 1.968, 1.9733333333333327, 1.9786666666666664, 1.984, 1.9893333333333327, 1.9946666666666664, 2.0, 2.0053333333333327, 2.0106666666666664, 2.016, 2.0213333333333328, 2.0266666666666664, 2.032, 2.0373333333333328, 2.0426666666666664, 2.048, 2.053333333333333, 2.0586666666666664, 2.064, 2.069333333333333, 2.0746666666666664, 2.08, 2.085333333333333, 2.0906666666666665, 2.096, 2.101333333333333, 2.1066666666666665, 2.112, 2.117333333333333, 2.1226666666666665, 2.128, 2.133333333333333, 2.1386666666666665, 2.144, 2.149333333333333, 2.1546666666666665, 2.16, 2.165333333333333, 2.1706666666666665, 2.176, 2.181333333333333, 2.1866666666666665, 2.192, 2.197333333333333, 2.2026666666666666, 2.208, 2.213333333333333, 2.2186666666666666, 2.224, 2.229333333333333, 2.2346666666666666, 2.24, 2.245333333333333, 2.2506666666666666, 2.2560000000000002, 2.261333333333333, 2.2666666666666666, 2.2719999999999994, 2.277333333333333, 2.2826666666666666, 2.2879999999999994, 2.293333333333333, 2.2986666666666666, 2.3039999999999994, 2.309333333333333, 2.3146666666666667, 2.3199999999999994, 2.325333333333333, 2.3306666666666667, 2.3359999999999994, 2.341333333333333, 2.3466666666666667, 2.3519999999999994, 2.357333333333333, 2.3626666666666667, 2.3679999999999994, 2.373333333333333, 2.3786666666666667, 2.3839999999999995, 2.389333333333333, 2.3946666666666667, 2.3999999999999995, 2.405333333333333, 2.4106666666666667, 2.4159999999999995, 2.421333333333333, 2.4266666666666667, 2.4319999999999995, 2.437333333333333, 2.4426666666666668, 2.4479999999999995, 2.453333333333333, 2.458666666666667, 2.4639999999999995, 2.469333333333333, 2.474666666666667, 2.4799999999999995, 2.485333333333333, 2.490666666666667, 2.4959999999999996, 2.501333333333333, 2.506666666666667, 2.5119999999999996, 2.517333333333333, 2.522666666666667, 2.5279999999999996, 2.533333333333333, 2.538666666666667, 2.5439999999999996, 2.5493333333333332, 2.554666666666667, 2.5599999999999996, 2.5653333333333332, 2.570666666666667, 2.5759999999999996, 2.5813333333333333, 2.586666666666666, 2.5919999999999996, 2.5973333333333333, 2.602666666666666, 2.6079999999999997, 2.6133333333333333, 2.618666666666666, 2.6239999999999997, 2.6293333333333333, 2.634666666666666, 2.6399999999999997, 2.6453333333333333, 2.650666666666666, 2.6559999999999997, 2.6613333333333333, 2.666666666666666, 2.6719999999999997, 2.6773333333333333, 2.682666666666666, 2.6879999999999997, 2.6933333333333334, 2.698666666666666, 2.7039999999999997, 2.7093333333333334, 2.714666666666666, 2.7199999999999998, 2.7253333333333334, 2.730666666666666, 2.7359999999999998, 2.7413333333333334, 2.746666666666666, 2.752, 2.7573333333333334, 2.762666666666666, 2.768, 2.7733333333333334, 2.778666666666666, 2.784, 2.7893333333333334, 2.794666666666666, 2.8, 2.8053333333333335, 2.810666666666666, 2.816, 2.8213333333333335, 2.826666666666666, 2.832, 2.8373333333333335, 2.8426666666666662, 2.848, 2.8533333333333335, 2.8586666666666662, 2.864, 2.8693333333333335, 2.8746666666666663, 2.88, 2.8853333333333335, 2.8906666666666663, 2.896, 2.9013333333333335, 2.9066666666666663, 2.912, 2.9173333333333327, 2.9226666666666663, 2.928, 2.9333333333333327, 2.9386666666666663, 2.944, 2.9493333333333327, 2.9546666666666663, 2.96, 2.9653333333333327, 2.9706666666666663, 2.976, 2.9813333333333327, 2.9866666666666664, 2.992, 2.9973333333333327, 3.0026666666666664, 3.008, 3.0133333333333328, 3.0186666666666664, 3.024, 3.0293333333333328, 3.0346666666666664, 3.04, 3.045333333333333, 3.0506666666666664, 3.056, 3.061333333333333, 3.0666666666666664, 3.072, 3.077333333333333, 3.0826666666666664, 3.088, 3.093333333333333, 3.0986666666666665, 3.104, 3.109333333333333, 3.1146666666666665, 3.12, 3.125333333333333, 3.1306666666666665, 3.136, 3.141333333333333, 3.1466666666666665, 3.152, 3.157333333333333, 3.1626666666666665, 3.168, 3.173333333333333, 3.1786666666666665, 3.184, 3.189333333333333, 3.1946666666666665, 3.2, 3.205333333333333, 3.2106666666666666, 3.216, 3.221333333333333, 3.2266666666666666, 3.2319999999999993, 3.237333333333333, 3.2426666666666666, 3.2479999999999993, 3.253333333333333, 3.2586666666666666, 3.2639999999999993, 3.269333333333333, 3.2746666666666666, 3.2799999999999994, 3.285333333333333, 3.2906666666666666, 3.2959999999999994, 3.301333333333333, 3.3066666666666666, 3.3119999999999994, 3.317333333333333, 3.3226666666666667, 3.3279999999999994, 3.333333333333333, 3.3386666666666667, 3.3439999999999994, 3.349333333333333, 3.3546666666666667, 3.3599999999999994, 3.365333333333333, 3.3706666666666667, 3.3759999999999994, 3.381333333333333, 3.3866666666666667, 3.3919999999999995, 3.397333333333333, 3.4026666666666667, 3.4079999999999995, 3.413333333333333, 3.4186666666666667, 3.4239999999999995, 3.429333333333333, 3.4346666666666668, 3.4399999999999995, 3.445333333333333, 3.4506666666666668, 3.4559999999999995, 3.461333333333333, 3.466666666666667, 3.4719999999999995, 3.477333333333333, 3.482666666666667, 3.4879999999999995, 3.493333333333333, 3.498666666666667, 3.5039999999999996, 3.509333333333333, 3.514666666666667, 3.5199999999999996, 3.525333333333333, 3.530666666666667, 3.5359999999999996, 3.541333333333333, 3.546666666666667, 3.5519999999999996, 3.5573333333333332, 3.562666666666666, 3.5679999999999996, 3.5733333333333333, 3.578666666666666, 3.5839999999999996, 3.5893333333333333, 3.594666666666666, 3.5999999999999996, 3.6053333333333333, 3.610666666666666, 3.6159999999999997, 3.6213333333333333, 3.626666666666666, 3.6319999999999997, 3.6373333333333333, 3.642666666666666, 3.6479999999999997, 3.6533333333333333, 3.658666666666666, 3.6639999999999997, 3.6693333333333333, 3.674666666666666, 3.6799999999999997, 3.6853333333333333, 3.690666666666666, 3.6959999999999997, 3.7013333333333334, 3.706666666666666, 3.7119999999999997, 3.7173333333333334, 3.722666666666666, 3.7279999999999998, 3.7333333333333334, 3.738666666666666, 3.7439999999999998, 3.7493333333333334, 3.754666666666666, 3.76, 3.7653333333333334, 3.770666666666666, 3.776, 3.7813333333333334, 3.786666666666666, 3.792, 3.7973333333333334, 3.802666666666666, 3.808, 3.8133333333333335, 3.818666666666666, 3.824, 3.8293333333333335, 3.8346666666666662, 3.84, 3.8453333333333335, 3.8506666666666662, 3.856, 3.8613333333333335, 3.8666666666666663, 3.872, 3.8773333333333326, 3.8826666666666663, 3.888, 3.8933333333333326, 3.8986666666666663, 3.904, 3.9093333333333327, 3.9146666666666663, 3.92, 3.9253333333333327, 3.9306666666666663, 3.936, 3.9413333333333327, 3.9466666666666663, 3.952, 3.9573333333333327, 3.9626666666666663, 3.968, 3.9733333333333327, 3.9786666666666664, 3.984, 3.9893333333333327, 3.9946666666666664, 4.0, ])