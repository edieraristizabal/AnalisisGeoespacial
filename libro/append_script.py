import os

bib_entries = """
@article{Bates_2008,
  author = {Bates, B. C. and Kundzewicz, Z. W. and Wu, S. and Palutikof, J. P.},
  title = {Climate change and water. Technical paper of the Intergovernmental Panel on Climate Change},
  journal = {IPCC Secretariat},
  year = {2008},
  address = {Geneva}
}

@article{Dottori_2018,
  author = {Dottori, F. and Szewczyk, W. and Ciscar, J. C. and Zhao, F. and Alfieri, L. and Hirabayashi, Y. and Bianchi, A. and Mongelli, I. and Frieler, K. and Betts, R. A. and Feyen, L.},
  title = {Increased human and economic losses from river floods with global warming},
  journal = {Nature Climate Change},
  volume = {8},
  pages = {781--786},
  year = {2018},
  doi = {10.1038/s41558-018-0257-z}
}

@article{AghaKouchak_2020,
  author = {AghaKouchak, A. and Chiang, F. and Huning, L. S. and Love, C. A. and Mallakpour, I. and Mazdiyasni, O. and Moftakhari, H. and Papalexiou, S. M. and Zscheischler, J. and Vahedifard, F.},
  title = {Climate Extremes and Compound Hazards in a Warming World},
  journal = {Annual Review of Earth and Planetary Sciences},
  volume = {48},
  pages = {519--548},
  year = {2020},
  doi = {10.1146/annurev-earth-071719-055228}
}

@article{Meehl_2000,
  author = {Meehl, G. A. and Zwiers, F. and Evans, J. and Knutson, T. and Mearns, L. and Whetton, P.},
  title = {Trends in extreme weather and climate events: Issues related to modeling extremes in projections of future climate change},
  journal = {Bulletin of the American Meteorological Society},
  volume = {81},
  pages = {427--436},
  year = {2000},
  doi = {10.1175/1520-0477(2000)081<0427:TIEWAC>2.3.CO;2}
}

@article{Rojas_2013,
  author = {Rojas, R. and Feyen, L. and Watkiss, P.},
  title = {Climate change and river floods in the European Union: Socio-economic consequences and the costs and benefits of adaptation},
  journal = {Global Environmental Change},
  volume = {23},
  pages = {1737--1751},
  year = {2013},
  doi = {10.1016/j.gloenvcha.2013.08.006}
}

@article{Westra_2014,
  author = {Westra, S. and Fowler, H. J. and Evans, J. P. and Alexander, L. V. and Berg, P. and Johnson, F. and Kendon, E. J. and Lenderink, G. and Roberts, N. M.},
  title = {Future changes to the intensity and frequency of short-duration extreme rainfall},
  journal = {Reviews of Geophysics},
  volume = {52},
  pages = {522--555},
  year = {2014},
  doi = {10.1002/2014RG000464}
}

@article{Wilby_2008,
  author = {Wilby, R. L. and Dessai, S.},
  title = {Robust adaptation to climate change},
  journal = {Weather},
  volume = {65},
  pages = {180--185},
  year = {2010}
}
@article{Gelfan_2017,
  author = {Gelfan, A. and Gustafsson, D. and Motovilov, Y. and Arheimer, B. and Kalugin, A. and Krylenko, I. and Lavrenov, A.},
  title = {Climate change impact on the water regime of two great Arctic rivers: modeling and uncertainty issues},
  journal = {Climatic Change},
  volume = {141},
  pages = {499--515},
  year = {2017},
  doi = {10.1007/s10584-016-1710-5}
}

@article{Milly_2005,
  author = {Milly, P. C. D. and Dunne, K. A. and Vecchia, A. V.},
  title = {Global pattern of trends in streamflow and water availability in a changing climate},
  journal = {Nature},
  volume = {438},
  pages = {347--350},
  year = {2005},
  doi = {10.1038/nature04312}
}

@article{Gourley_2013,
  author = {Gourley, J. J. and Hong, Y. and Flamig, Z. L. and Arthur, A. and Clark, R. and Calianese, M. and ... and Krajewski, W. F.},
  title = {A unified flash flood database across the United States},
  journal = {Bulletin of the American Meteorological Society},
  volume = {94},
  pages = {799--805},
  year = {2013},
  doi = {10.1175/BAMS-D-12-00198.1}
}

@article{Bouwer_2011,
  author = {Bouwer, L. M.},
  title = {Have disaster losses increased due to anthropogenic climate change?},
  journal = {Bulletin of the American Meteorological Society},
  volume = {92},
  pages = {39--46},
  year = {2011},
  doi = {10.1175/2010BAMS3092.1}
}

@article{Kundzewicz_2014,
  author = {Kundzewicz, Z. W. and Kanae, S. and Seneviratne, S. I. and Handmer, J. and Nicholls, N. and Peduzzi, P. and ... and Sherstyukov, B.},
  title = {Flood risk and climate change: global and regional perspectives},
  journal = {Hydrological Sciences Journal},
  volume = {59},
  pages = {1--28},
  year = {2014},
  doi = {10.1080/02626667.2013.857411}
}

@article{Madsen_2014,
  author = {Madsen, H. and Lawrence, D. and Lang, M. and Martinkova, M. and Kjeldsen, T. R.},
  title = {Review of trend analysis and climate change projections of extreme precipitation and river flood frequency in Europe},
  journal = {Journal of Hydrology},
  volume = {519},
  pages = {3634--3650},
  year = {2014},
  doi = {10.1016/j.jhydrol.2014.11.003}
}

@article{Trenberth_2014,
  author = {Trenberth, K. E. and Dai, A. and van der Schrier, G. and Jones, P. D. and Barichivich, J. and Briffa, K. R. and Sheffield, J.},
  title = {Global warming and changes in drought},
  journal = {Nature Climate Change},
  volume = {4},
  pages = {17--22},
  year = {2014},
  doi = {10.1038/nclimate2067}
}

@article{Coumou_2012,
  author = {Coumou, D. and Rahmstorf, S.},
  title = {A decade of weather extremes},
  journal = {Nature Climate Change},
  volume = {2},
  pages = {491--496},
  year = {2012},
  doi = {10.1038/nclimate1452}
}

@article{Allen_2002,
  author = {Allen, M. R. and Ingram, W. J.},
  title = {Constraints on future changes in climate and the hydrologic cycle},
  journal = {Nature},
  volume = {419},
  pages = {224--232},
  year = {2002},
  doi = {10.1038/nature01092}
}

@article{Alexander_2006,
  author = {Alexander, L. V. and Zhang, X. and Peterson, T. C. and Caesar, J. and Gleason, B. and Klein Tank, A. M. G. and ... and Tagipour, A.},
  title = {Global observed changes in daily climate extremes of temperature and precipitation},
  journal = {Journal of Geophysical Research: Atmospheres},
  volume = {111},
  pages = {D05109},
  year = {2006},
  doi = {10.1029/2005JD006290}
}

@article{Bates_2008_IPCC,
  author = {Bates, B. C. and Kundzewicz, Z. W. and Wu, S. and Palutikof, J. P.},
  title = {Climate Change and Water},
  journal = {IPCC Technical Paper VI. Geneva: IPCC Secretariat},
  year = {2008}
}

@article{Min_2011,
  author = {Min, S. K. and Zhang, X. and Zwiers, F. W. and Hegerl, G. C.},
  title = {Human contribution to more-intense precipitation extremes},
  journal = {Nature},
  volume = {470},
  pages = {378--381},
  year = {2011},
  doi = {10.1038/nature09763}
}

@article{Pall_2011,
  author = {Pall, P. and Aina, T. and Stone, D. A. and Stott, P. A. and Nozawa, T. and Hilberts, A. G. J. and ... and Allen, M. R.},
  title = {Anthropogenic greenhouse gas contribution to flood risk in England and Wales in autumn 2000},
  journal = {Nature},
  volume = {470},
  pages = {382--385},
  year = {2011},
  doi = {10.1038/nature09762}
}

@article{Jongman_2012,
  author = {Jongman, B. and Ward, P. J. and Aerts, J. C. J. H.},
  title = {Global exposure to river and coastal flooding: Long term trends and changes},
  journal = {Global Environmental Change},
  volume = {22},
  pages = {823--835},
  year = {2012},
  doi = {10.1016/j.gloenvcha.2012.07.004}
}

@article{Mirza_2003,
  author = {Mirza, M. M. Q.},
  title = {Climate change and extreme weather events: can developing countries adapt?},
  journal = {Climate Policy},
  volume = {3},
  pages = {233--248},
  year = {2003},
  doi = {10.1016/S1469-3062(03)00052-4}
}

@article{Hall_2014,
  author = {Hall, J. and Arheimer, B. and Borga, M. and Br{\'o}nneke, R. and Chirico, P. and Corsini, S. and ... and Zehe, E.},
  title = {Understanding flood regime changes in Europe: a state-of-the-art assessment},
  journal = {Hydrology and Earth System Sciences},
  volume = {18},
  pages = {2735--2772},
  year = {2014},
  doi = {10.5194/hess-18-2735-2014}
}

@article{Kharin_2013,
  author = {Kharin, V. V. and Zwiers, F. W. and Zhang, X. and Wehner, M.},
  title = {Changes in temperature and precipitation extremes in the CMIP5 ensemble},
  journal = {Climatic Change},
  volume = {119},
  pages = {345--357},
  year = {2013},
  doi = {10.1007/s10584-013-0705-8}
}

@article{Schreider_2000,
  author = {Schreider, S. Y. and Smith, D. I. and Jakeman, A. J.},
  title = {Climate change impacts on urban flooding},
  journal = {Climatic Change},
  volume = {47},
  pages = {91--115},
  year = {2000},
  doi = {10.1023/A:1005621523177}
}

@article{Oki_2006,
  author = {Oki, T. and Kanae, S.},
  title = {Global hydrological cycles and world water resources},
  journal = {Science},
  volume = {313},
  pages = {1068--1072},
  year = {2006},
  doi = {10.1126/science.1128845}
}

@article{Fowler_2007,
  author = {Fowler, H. J. and Blenkinsop, S. and Tebaldi, C.},
  title = {Linking climate change modelling to impacts studies: recent advances in downscaling techniques for hydrological modelling},
  journal = {International Journal of Climatology},
  volume = {27},
  pages = {1547--1578},
  year = {2007},
  doi = {10.1002/joc.1556}
}

@article{Wetter_2011,
  author = {Wetter, O. and Pfister, C. and Weingartner, R. and Luterbacher, J. and Reist, T. and Tr{\"o}sch, J.},
  title = {The largest floods in the High Rhine basin since 1268 assessed from documentary and instrumental evidence},
  journal = {Hydrological Sciences Journal},
  volume = {56},
  pages = {733--758},
  year = {2011},
  doi = {10.1080/02626667.2011.583613}
}

@article{Gaume_2009,
  author = {Gaume, E. and Bain, V. and Bernardara, P. and Newinger, O. and Barbuc, M. and Bateman, A. and ... and Viglione, A.},
  title = {A compilation of data on European flash floods},
  journal = {Journal of Hydrology},
  volume = {367},
  pages = {70--78},
  year = {2009},
  doi = {10.1016/j.jhydrol.2008.12.028}
}

@article{Zhang_2007,
  author = {Zhang, X. and Zwiers, F. W. and Hegerl, G. C. and Lambert, F. H. and Gillett, N. P. and Solomon, S. and Stott, P. A. and Nozawa, T.},
  title = {Detection of human influence on twentieth-century precipitation trends},
  journal = {Nature},
  volume = {448},
  pages = {461--465},
  year = {2007},
  doi = {10.1038/nature06025}
}

@article{Seneviratne_2012,
  author = {Seneviratne, S. I. and Nicholls, N. and Easterling, D. and Goodess, C. M. and Kanae, S. and Kossin, J. and ... and Zhang, X.},
  title = {Changes in climate extremes and their impacts on the natural physical environment},
  journal = {Managing the Risks of Extreme Events and Disasters to Advance Climate Change Adaptation},
  pages = {109--230},
  year = {2012},
  publisher = {Cambridge University Press}
}

@article{Sillmann_2013,
  author = {Sillmann, J. and Kharin, V. V. and Zhang, X. and Zwiers, F. W. and Bronaugh, D.},
  title = {Climate extremes indices in the CMIP5 multimodel ensemble: Part 1. Model evaluation in the present climate},
  journal = {Journal of Geophysical Research: Atmospheres},
  volume = {118},
  pages = {1716--1733},
  year = {2013},
  doi = {10.1002/jgrd.50203}
}

@article{Milly_2008,
  author = {Milly, P. C. D. and Betancourt, J. and Falkenmark, M. and Hirsch, R. M. and Kundzewicz, Z. W. and Lettenmaier, D. P. and Stouffer, R. J.},
  title = {Stationarity is dead: Whither water management?},
  journal = {Science},
  volume = {319},
  pages = {573--574},
  year = {2008},
  doi = {10.1126/science.1151915}
}

@article{Gosling_2014,
  author = {Gosling, S. N. and Arnell, N. W.},
  title = {A global assessment of the impact of climate change on water scarcity},
  journal = {Climatic Change},
  volume = {134},
  pages = {371--385},
  year = {2016},
  doi = {10.1007/s10584-013-0853-x}
}

@article{Bloschl_2017,
  author = {Bl{\"o}schl, G. and Hall, J. and Parajka, J. and Perdig{\~a}o, R. A. P. and Merz, B. and Arheimer, B. and ... and {\v{Z}}ivkovi{\'c}, N.},
  title = {Changing climate shifts timing of European floods},
  journal = {Science},
  volume = {357},
  pages = {588--590},
  year = {2017},
  doi = {10.1126/science.aan2506}
}

@article{Lenderink_2008,
  author = {Lenderink, G. and Van Meijgaard, E.},
  title = {Increase in hourly precipitation extremes beyond expectations from temperature changes},
  journal = {Nature Geoscience},
  volume = {1},
  pages = {511--514},
  year = {2008},
  doi = {10.1038/ngeo262}
}

@article{Piao_2010,
  author = {Piao, S. and Ciais, P. and Huang, Y. and Shen, Z. and Peng, S. and Li, J. and ... and Fang, J.},
  title = {The impacts of climate change on water resources and agriculture in China},
  journal = {Nature},
  volume = {467},
  pages = {43--51},
  year = {2010},
  doi = {10.1038/nature09364}
}

@article{Villarini_2013,
  author = {Villarini, G. and Smith, J. A. and Vecchi, G. A.},
  title = {Changing Frequency of Heavy Rainfall over the Central United States},
  journal = {Journal of Climate},
  volume = {26},
  pages = {351--357},
  year = {2013},
  doi = {10.1175/JCLI-D-12-00043.1}
}

@article{Knox_2000,
  author = {Knox, J. C.},
  title = {Sensitivity of modern and Holocene floods to climate change},
  journal = {Quaternary Science Reviews},
  volume = {19},
  pages = {439--457},
  year = {2000},
  doi = {10.1016/S0277-3791(99)00074-8}
}

@article{Groisman_2005,
  author = {Groisman, P. Y. and Knight, R. W. and Easterling, D. R. and Karl, T. R. and Hegerl, G. C. and Razuvaev, V. N.},
  title = {Trends in intense precipitation in the climate record},
  journal = {Journal of Climate},
  volume = {18},
  pages = {1326--1350},
  year = {2005},
  doi = {10.1175/JCLI3339.1}
}

@article{Schmocker_Finkel_2010,
  author = {Schmocker-Finkel, J. and Naef, F.},
  title = {More frequent flooding? Changes in flood frequency in Switzerland since 1850},
  journal = {Journal of Hydrology},
  volume = {381},
  pages = {1--8},
  year = {2010},
  doi = {10.1016/j.jhydrol.2009.11.011}
}

@article{Arnell_2004,
  author = {Arnell, N. W.},
  title = {Climate change and global water resources: SRES emissions and socio-economic scenarios},
  journal = {Global Environmental Change},
  volume = {14},
  pages = {31--52},
  year = {2004},
  doi = {10.1016/j.gloenvcha.2003.10.006}
}

@article{Dankers_2014,
  author = {Dankers, R. and Arnell, N. W. and Clark, D. B. and Falloon, P. D. and Fekete, B. M. and Gosling, S. N. and ... and Wada, Y.},
  title = {First look at changes in flood hazard in the Inter-Sectoral Impact Model Intercomparison Project ensemble},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {111},
  pages = {3257--3261},
  year = {2014},
  doi = {10.1073/pnas.1302078110}
}

@article{Kysel_2012,
  author = {Kysel{\\'y}, J. and Beguer{\'i}a, S. and Beranova, R. and Ga{\'a}l, L. and Lopez-Moreno, J. I.},
  title = {Spatial patterns and underlying mechanisms of winter extreme precipitation and circulation connections over Europe},
  journal = {International Journal of Climatology},
  volume = {32},
  pages = {1602--1621},
  year = {2012},
  doi = {10.1002/joc.2376}
}

@article{Ward_2013,
  author = {Ward, P. J. and Jongman, B. and Weiland, F. S. and Bouwman, A. and van Beek, R. and Bijlsma, W. and ... and Winsemius, H. C.},
  title = {Assessing flood risk at the global scale: model setup, results, and sensitivity},
  journal = {Environmental Research Letters},
  volume = {8},
  pages = {044019},
  year = {2013},
  doi = {10.1088/1748-9326/8/4/044019}
}

@article{Petrow_2009,
  author = {Petrow, T. and Merz, B.},
  title = {Trends in flood magnitude, frequency and seasonality in Germany in the period 1951--2002},
  journal = {Journal of Hydrology},
  volume = {371},
  pages = {129--141},
  year = {2009},
  doi = {10.1016/j.jhydrol.2009.03.024}
}

@article{Schollaertuz_2004,
  author = {Schollaer-Tuz, S. and et al.},
  title = {Variability of floods in changing climates},
  journal = {Hydrological Processes},
  year = {2004}
}

@article{Vogel_2011,
  author = {Vogel, R. M. and Yaindl, C. and Walter, M.},
  title = {Nonstationarity: Flood magnification and yield patterns in the United States},
  journal = {Journal of the American Water Resources Association},
  volume = {47},
  pages = {464--474},
  year = {2011},
  doi = {10.1111/j.1752-1688.2011.00541.x}
}

@article{Prein_2017,
  author = {Prein, A. F. and Rasmussen, R. M. and Ikeda, K. and Liu, C. and Clark, M. P. and Holland, G. J.},
  title = {The future intensification of hourly precipitation extremes},
  journal = {Nature Climate Change},
  volume = {7},
  pages = {48--52},
  year = {2017},
  doi = {10.1038/nclimate3168}
}

@article{Milly_2015,
  author = {Milly, P. C. D. and Dunne, K. A. and Betancourt, J. and Falkenmark, M. and Hirsch, R. M. and Kundzewicz, Z. W. and Lettenmaier, D. P. and Stouffer, R. J.},
  title = {Stationarity is dead: Whither water management?},
  journal = {Science},
  year = {2015}
}

@article{Guhathakurta_2011,
  author = {Guhathakurta, P. and Sreejith, O. P. and Menon, P. A.},
  title = {Impact of climate change on extreme rainfall events and flood risk in India},
  journal = {Journal of Earth System Science},
  volume = {120},
  pages = {359--373},
  year = {2011},
  doi = {10.1007/s12040-011-0082-5}
}

@article{Hirabayashi_2013,
  author = {Hirabayashi, Y. and Mahendran, R. and Koirala, S. and Konoshima, L. and Yamazaki, D. and Watanabe, S. and Kim, H. and Kanae, S.},
  title = {Global flood risk under climate change},
  journal = {Nature Climate Change},
  volume = {3},
  pages = {816--821},
  year = {2013},
  doi = {10.1038/nclimate1911}
}

@article{Milly_2002,
  author = {Milly, P. C. D. and Wetherald, R. T. and Dunne, K. A. and Delworth, T. L.},
  title = {Increasing risk of great floods in a changing climate},
  journal = {Nature},
  volume = {415},
  pages = {514--517},
  year = {2002},
  doi = {10.1038/415514a}
}

@article{Zhang_2018,
  author = {Zhang, D. and et al.},
  title = {High-resolution ensemble projections and uncertainty assessment of regional climate change over China},
  journal = {Hydrology and Earth System Sciences},
  volume = {22},
  pages = {521--540},
  year = {2018}
}

@article{O_Gorman_2015,
  author = {O'Gorman, P. A.},
  title = {Precipitation Extremes Under Climate Change},
  journal = {Current Climate Change Reports},
  volume = {1},
  pages = {49--59},
  year = {2015},
  doi = {10.1007/s40641-015-0009-3}
}

@article{Pielke_2000,
  author = {Pielke, R. A. and Downton, M. W.},
  title = {Precipitation and Damaging Floods: Trends in the United States, 1932--97},
  journal = {Journal of Climate},
  volume = {13},
  pages = {3625--3637},
  year = {2000}
}

@article{Kundzewicz_2005,
  author = {Kundzewicz, Z. W. and Radziejewski, M. and Pi{\'n}skwar, I.},
  title = {Precipitation extremes in the changing climate of Europe},
  journal = {Climate Research},
  volume = {31},
  pages = {51--58},
  year = {2005}
}

@article{Christensen_2007,
  author = {Christensen, J. H. and Christensen, O. B.},
  title = {A summary of the PRUDENCE model projections of changes in European climate by the end of this century},
  journal = {Climatic Change},
  volume = {81},
  pages = {7--30},
  year = {2007}
}

@article{Donat_2016,
  author = {Donat, M. G. and Lowry, A. L. and Alexander, L. V. and O'Gorman, P. A. and Maher, N.},
  title = {More extreme precipitation in the world's dry and wet regions},
  journal = {Nature Climate Change},
  volume = {6},
  pages = {508--513},
  year = {2016},
  doi = {10.1038/nclimate2941}
}

@article{Arnone_2013,
  author = {Arnone, E. and Pumo, D. and Viola, F. and Noto, L. V. and La Loggia, G.},
  title = {Rainfall statistics changes in Sicily},
  journal = {Hydrology and Earth System Sciences},
  volume = {17},
  pages = {2449--2458},
  year = {2013},
  doi = {10.5194/hess-17-2449-2013}
}

@article{Moberg_2006,
  author = {Moberg, A. and Jones, P. D. and Lister, D. and Walther, A. and Brunet, M. and Jacobeit, J. and ... and Yang, S.},
  title = {Indices for daily temperature and precipitation extremes in Europe analyzed for the period 1901--2000},
  journal = {Journal of Geophysical Research: Atmospheres},
  volume = {111},
  pages = {D22106},
  year = {2006},
  doi = {10.1029/2006JD007103}
}

@article{Burn_2002,
  author = {Burn, D. H. and Hag Elnur, M. A.},
  title = {Detection of hydrologic trends and variability},
  journal = {Journal of Hydrology},
  volume = {255},
  pages = {107--122},
  year = {2002}
}

@article{Nakicenovic_2000,
  author = {Nakicenovic, N. and Swart, R.},
  title = {Special Report on Emissions Scenarios},
  journal = {A Special Report of Working Group III of the Intergovernmental Panel on Climate Change},
  year = {2000},
  publisher = {Cambridge University Press}
}

@article{Kunkel_2013,
  author = {Kunkel, K. E. and Karl, T. R. and Brooks, H. and Kossin, J. and Lawrimore, J. H. and Arndt, D. and ... and Yin, D.},
  title = {Monitoring and Understanding Trends in Extreme Storms: State of Knowledge},
  journal = {Bulletin of the American Meteorological Society},
  volume = {94},
  pages = {499--514},
  year = {2013},
  doi = {10.1175/BAMS-D-11-00262.1}
}

@article{Bates_2008_IPCC_Water,
  author = {Bates, B. C. and Kundzewicz, Z. W. and Wu, S. and Palutikof, J. P.},
  title = {Climate Change and Water: IPCC Technical Paper VI},
  journal = {Intergovernmental Panel on Climate Change},
  year = {2008}
}

@article{Tramblay_2012,
  author = {Tramblay, Y. and Badi, W. and Driouech, F. and El Adlouni, S. and Neppel, L. and Servat, E.},
  title = {Climate change impacts on extreme precipitation in Morocco},
  journal = {Global and Planetary Change},
  volume = {82-83},
  pages = {104--114},
  year = {2012},
  doi = {10.1016/j.gloplacha.2011.12.002}
}

@article{Zscheischler_2020,
  author = {Zscheischler, J. and Martius, O. and Westra, S. and Bevacqua, E. and Raymond, C. and Horton, R. M. and ... and Vignotto, E.},
  title = {A typology of compound weather and climate events},
  journal = {Nature Reviews Earth & Environment},
  volume = {1},
  pages = {333--347},
  year = {2020},
  doi = {10.1038/s43017-020-0060-z}
}

@article{Katz_2002,
  author = {Katz, R. W. and Parlange, M. B. and Naveau, P.},
  title = {Statistics of extremes in hydrology},
  journal = {Advances in Water Resources},
  volume = {25},
  pages = {1287--1304},
  year = {2002},
  doi = {10.1016/S0309-1708(02)00056-8}
}

@article{Li_2016,
  author = {Li, Z. and et al.},
  title = {Hierarchy evaluation of water resources vulnerability under climate change in Beijing, China},
  journal = {Hydrological Processes},
  year = {2016}
}

@article{Winsemius_2016,
  author = {Winsemius, H. C. and Aerts, J. C. J. H. and van Beek, L. P. H. and Bierkens, M. F. P. and Bouwman, A. and Jongman, B. and Kwadijk, J. C. J. and Ligtvoet, W. and Lucas, P. L. and van Vuuren, D. P. and Ward, P. J.},
  title = {Global drivers of future river flood risk},
  journal = {Nature Climate Change},
  volume = {6},
  pages = {381--385},
  year = {2016},
  doi = {10.1038/nclimate2893}
}

@article{Blenkinsop_2018,
  author = {Blenkinsop, S. and Fowler, H. J. and Barbero, R. and Chan, S. C. and Guerreiro, S. B. and Kendon, E. and Lenderink, G. and Lewis, E. and Li, X. F. and Westra, S. and Alexander, L. and Allan, R. P. and Berg, P. and Dunn, R. J. H. and Ekstr{\"o}m, M. and Evans, J. P. and Holland, G. and Jones, R. and Kjellstr{\"o}m, E. and Klein-Tank, A. and Lettenmaier, D. and Mishra, V. and Prein, A. F. and Sheffield, J. and Tye, M. R.},
  title = {The INTENSE project: Using observations and models to understand the past, present and future of sub-daily rainfall extremes},
  journal = {Advances in Science and Research},
  volume = {15},
  pages = {117--126},
  year = {2018},
  doi = {10.5194/asr-15-117-2018}
}

@article{Douglas_2000,
  author = {Douglas, E. M. and Vogel, R. M. and Kroll, C. N.},
  title = {Trends in floods and low flows in the United States: impact of spatial correlation},
  journal = {Journal of Hydrology},
  volume = {240},
  pages = {90--105},
  year = {2000},
  doi = {10.1016/S0022-1694(00)00336-X}
}

@article{Tebaldi_2006,
  author = {Tebaldi, C. and Hayhoe, K. and Arblaster, J. M. and Meehl, G. A.},
  title = {Going to the Extremes},
  journal = {Climatic Change},
  volume = {79},
  pages = {185--211},
  year = {2006},
  doi = {10.1007/s10584-006-9051-4}
}

@article{Meehl_2007,
  author = {Meehl, G. A. and Stocker, T. F. and Collins, W. D. and Friedlingstein, P. and Gaye, A. T. and Gregory, J. M. and ... and Zhao, Z. C.},
  title = {Global Climate Projections},
  journal = {In: Climate Change 2007: The Physical Science Basis},
  year = {2007}
}

@article{Katz_2005,
  author = {Katz, R. W. and Brush, G. S. and Parlange, M. B.},
  title = {Statistics of extremes: Modeling ecological disturbances},
  journal = {Ecology},
  volume = {86},
  pages = {1124--1134},
  year = {2005},
  doi = {10.1890/04-0606}
}

@article{Field_2012,
  author = {Field, C. B. and Barros, V. and Stocker, T. F. and Qin, D. and Dokken, D. J. and Ebi, K. L. and ... and Midgley, P. M.},
  title = {Managing the Risks of Extreme Events and Disasters to Advance Climate Change Adaptation},
  journal = {A Special Report of Working Groups I and II of the Intergovernmental Panel on Climate Change},
  year = {2012},
  publisher = {Cambridge University Press}
}

@article{Robson_2002,
  author = {Robson, A. J.},
  title = {Evidence for trends in UK flooding},
  journal = {Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences},
  volume = {360},
  pages = {1327--1343},
  year = {2002},
  doi = {10.1098/rsta.2002.1003}
}

@article{Gu_2012,
  author = {Gu, H. and Yu, Z. and Wang, G. and Wang, J. and Ju, Q. and Yang, C. and Fan, C.},
  title = {Impact of climate change on hydrological extremes in the Yangtze River Basin, China},
  journal = {Stochastic Environmental Research and Risk Assessment},
  volume = {26},
  pages = {1087--1097},
  year = {2012},
  doi = {10.1007/s00477-012-0576-9}
}

@article{Alfieri_2015,
  author = {Alfieri, L. and Burek, P. and Feyen, L. and Forzieri, G.},
  title = {Global warming increases the frequency of river floods in Europe},
  journal = {Hydrology and Earth System Sciences},
  volume = {19},
  pages = {2247--2260},
  year = {2015},
  doi = {10.5194/hess-19-2247-2015}
}

@article{Stocker_2013,
  author = {Stocker, T. F. and Qin, D. and Plattner, G.-K. and Tignor, M. and Allen, S. K. and Boschung, J. and Nauels, A. and Xia, Y. and Bex, V. and Midgley, P. M.},
  title = {Climate Change 2013: The Physical Science Basis. Contribution of Working Group I to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change},
  journal = {IPCC},
  year = {2013},
  publisher = {Cambridge University Press}
}

@article{Froude_2018,
  author = {Froude, M. J. and Petley, D. N.},
  title = {Global fatal landslide occurrence from 2004 to 2016},
  journal = {Natural Hazards and Earth System Sciences},
  volume = {18},
  pages = {2161--2181},
  year = {2018},
  doi = {10.5194/nhess-18-2161-2018}
}

@article{Pall_2007,
  author = {Pall, P. and Allen, M. R. and Stone, D. A.},
  title = {Testing the Clausius-Clapeyron constraint on changes in extreme precipitation under CO2 warming},
  journal = {Climate Dynamics},
  volume = {28},
  pages = {351--363},
  year = {2007},
  doi = {10.1007/s00382-006-0180-2}
}

@article{Lenderink_2010,
  author = {Lenderink, G. and Van Meijgaard, E.},
  title = {Linking increases in hourly precipitation extremes to atmospheric temperature and moisture changes},
  journal = {Environmental Research Letters},
  volume = {5},
  pages = {025208},
  year = {2010},
  doi = {10.1088/1748-9326/5/2/025208}
}

@article{Milly_2002_Risk,
  author = {Milly, P. C. D. and Wetherald, R. T. and Dunne, K. A. and Delworth, T. L.},
  title = {Increasing risk of great floods in a changing climate},
  journal = {Nature},
  volume = {415},
  pages = {514--517},
  year = {2002},
  doi = {10.1038/415514a}
}

@article{van_Dijk_2013,
  author = {van Dijk, A. I. J. M. and Beck, H. E. and Crosbie, R. S. and de Jeu, R. A. M. and Liu, Y. Y. and Podger, G. M. and Timbal, B. and Viney, N. R.},
  title = {The Millennium Drought in southeast Australia (2001--2009): Natural and human causes and implications for water resources, ecosystems, economy, and society},
  journal = {Water Resources Research},
  volume = {49},
  pages = {1040--1057},
  year = {2013},
  doi = {10.1002/wrcr.20123}
}

@article{Karl_1998,
  author = {Karl, T. R. and Knight, R. W.},
  title = {Secular trends of precipitation amount, frequency, and intensity in the United States},
  journal = {Bulletin of the American Meteorological Society},
  volume = {79},
  pages = {231--241},
  year = {1998},
  doi = {10.1175/1520-0477(1998)079<0231:STOPAF>2.0.CO;2}
}

@article{Koutsoyiannis_2003,
  author = {Koutsoyiannis, D.},
  title = {Climate change, the Hurst phenomenon, and hydrological statistics},
  journal = {Hydrological Sciences Journal},
  volume = {48},
  pages = {3--24},
  year = {2003},
  doi = {10.1623/hysj.48.1.3.43481}
}

@article{Rummukainen_2012,
  author = {Rummukainen, M.},
  title = {Changes in climate and weather extremes in the 21st century},
  journal = {Wiley Interdisciplinary Reviews: Climate Change},
  volume = {3},
  pages = {115--129},
  year = {2012},
  doi = {10.1002/wcc.160}
}

@article{Westra_2013,
  author = {Westra, S. and Alexander, L. V. and Zwiers, F. W.},
  title = {Global Increasing Trends in Annual Maximum Daily Precipitation},
  journal = {Journal of Climate},
  volume = {26},
  pages = {3904--3918},
  year = {2013},
  doi = {10.1175/JCLI-D-12-00502.1}
}

@article{Papalexiou_2013,
  author = {Papalexiou, S. M. and Koutsoyiannis, D. and Makropoulos, C.},
  title = {How extreme is extreme? An assessment of daily rainfall distribution tails},
  journal = {Hydrology and Earth System Sciences},
  volume = {17},
  pages = {851--862},
  year = {2013},
  doi = {10.5194/hess-17-851-2013}
}
"""

with open("/home/edier/GoogleDrive/CATEDRA/ANALISISGEOESPACIAL/Libro_AnalisisGeoespacial/100_references.bib", "a") as f:
    f.write(bib_entries)

print("Added")
