from matplotlib import pyplot as plt


def call_data(fn, *dt):
    for d in dt:
        if d is None: return
    fn(*dt)


def plot_data(
        data=None,
        title=None,
        xlabel=None, ylabel=None,
        xlim=None, ylim=None,
        dat_fn=lambda ax, dat: ax.plot(dat),
        ax_fn=lambda ax: ax,
        plt_fn=lambda plt: plt.tight_layout(),
        ax=None,
):
    if ax is not None: _ax = ax
    else: _, _ax = plt.subplots()
    call_data(dat_fn, _ax, data)
    call_data(_ax.set_title, title)
    call_data(_ax.set_xlabel, xlabel), call_data(_ax.set_ylabel, ylabel)
    call_data(_ax.set_xlim, xlim), call_data(_ax.set_ylim, ylim)
    call_data(ax_fn, ax), call_data(plt_fn, plt)
    if ax is None: plt.show()
