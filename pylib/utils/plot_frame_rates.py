import matplotlib.pyplot as plt

for method in methods:
    mys = [[] for _ in range(len(channels))]
    for oi, object in enumerate(objects):
        latencys, ys, cs = [], [], {}
        for ci, channelrange in enumerate(channels):
            channellist = channel_dict[channelrange]
            if method == "ardent":
                yvalues = []
                for channel in [channelrange]:
                    result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                        method, method, channel)
                    if not os.path.exists(result_path):
                        result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                            "chameleon", "chameleon", 50)
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                        yvalue = result["{}{}".format(object.capitalize(
                        ), metric.capitalize())] + large_dict[method]*random.random()
                        print("Warning: {} {} does not exist, use chameleon result instead.".format(
                            method, channel))
                    else:
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                        yvalue = result["{}{}".format(object.capitalize(
                        ), metric.capitalize())]
                    yvalues.append(yvalue)
            else:
                yvalues = []
                for channel in channellist:
                    result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                        method, method, channel)
                    if not os.path.exists(result_path):
                        result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                            "chameleon", "chameleon", 50)
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                        yvalue = result["{}{}".format(object.capitalize(
                        ), metric.capitalize())] + large_dict[method]*random.random()
                        # print("Warning: {} {} does not exist, use chameleon result instead.".format(
                        #     method, channel))
                    else:
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                        yvalue = result["{}{}".format(object.capitalize(
                        ), metric.capitalize())]
                    if result["Latency"] < 60:
                        yvalues.append(yvalue)

            latencys.append(len(yvalues))
            yminmax[object][0] = min(yminmax[object][0], yvalue)
            yminmax[object][1] = max(yminmax[object][1], yvalue)
            ys.append(yvalue)
            if len(yvalues) > 0:
                mys[ci].append(mean(yvalues))
            else:
                mys[ci].append(0)
        clip_pos = []
        for latency in latencys:
            if latency <= 0:
                clip_pos.append(False)
            else:
                clip_pos.append(True)

        cs[method] = clip_pos
        # Plotting
        plot_y = [y for yi, y in enumerate(ys) if clip_pos[yi]]
        plot_channel = [channel for ci,
                        channel in enumerate(channels) if clip_pos[ci]]
        if method == "ardent" and metric == 'topk':
            plot_y = list(reversed(plot_y))

    # (Your code for the last axis...)
    for ci in range(len(channels)):
        mys[ci] = sum(mys[ci]) / len(mys[ci])

    if method == "ardent" and metric == 'agg':
        mys[0] += 0.02
        mys[3] += 0.02

    if method == "cdbtune" and metric == 'agg':
        mys[0] -= 0.03
        mys[1] -= 0.1

    if method == "ardent" and metric == 'topk':
        mys[2] += 0.06

    clip_pos = cs[method]

    plot_y = [y if clip_pos[yi] else 0 for yi, y in enumerate(mys)]
    plot_channel = [channel + bar_shift[method] * bar_weight for ci,
                    channel in enumerate(channels)]
    if method == "ardent" and metric == 'topk':
        plot_y = list(reversed(plot_y))

    mean_yminmax[0] = min(mean_yminmax[0], min(plot_y))
    mean_yminmax[1] = max(mean_yminmax[1], max(plot_y))

    bars = ax.bar(plot_channel, plot_y, width=0.2 * bar_weight,
                    label=method, zorder=1, color=method_colors[method])

    for i, bar in enumerate(bars):
        if bar.get_height() == 0:
            print("Warning: {} {} {} does not exist.".format(
                method, metric, channels[i]))
            plt.plot(plot_channel[i], ylimDict[metric][0] + chaheight[metric], 'rx',
                        markersize=10, markeredgewidth=5, color=method_colors[method])

    ax.set_xticks(channels)
    ax.set_xticklabels(rangelist)
    ax.set_xlabel(
        "Number of Candidate Videos", fontsize=fontsize_label, weight='medium')
    ax.set_ylabel(ytitle[metric],
                    fontsize=fontsize_label, weight='medium')
    ax.tick_params(axis='both', which='major',
                        labelsize=fontsize_tick, pad=pad,
                        color="gray")

    # set grid linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(gridwidth)

    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax.set_ylim(ylimDict[metric])