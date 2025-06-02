import sys
sys.path.append('/opt/root6/lib')
import ROOT
thres = ['5k', '10k']
hists = []
nbins = 100
qmin = 0
qmax = 50
for t in thres:
    h = ROOT.TH1F(f"hthres{t}", f"threshold {t}", nbins, qmin, qmax)
    h.SetDirectory(0)
    hists.append(h)
for t in thres:
    f = ROOT.TFile(f'nonoise_pid13_thres{t}_unipolar/many_muon_hits.root')
    tree = f.Get("mu_ndlar/hits")
    tree.Draw(f"totQ >> hthres{t}", "tindex == 0")
    hists[0].Print("all")

c = ROOT.TCanvas()
c.Draw()
hists[0].Draw()
for h in hists[1:]:
    h.Draw("SAME")
c.Print("thres_comp_unipolar_pid13.png")
