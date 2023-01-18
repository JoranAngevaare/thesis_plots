import thesis_plots


def test_make_rebin_false():
    thesis_plots.setup_plt()
    thesis_plots.PlotLambdaCDM().plot(make_rebin=False)
    thesis_plots.save_fig('planck_cdm')
