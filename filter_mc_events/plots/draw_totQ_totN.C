#include <iostream>
#include <string>


// static std::string data_path = "../track_data.root";

// static std::string mc_path = "../pgun_pid13/many_muon_hits.root";
// static std::string effq_path = "../pgun_pid13/many_muon_effq.root";
// static std::string mc_path = "../pgun_pid13_constR/many_muon_hits.root";
// static std::string effq_path = "../pgun_pid13_constR/many_muon_effq.root";
// static std::string mc_path = "../pgun_pid13_transformed/many_muon_hits.root";
// static std::string effq_path = "../pgun_pid13_transformed/many_muon_effq.root";
// static std::string mc_path = "../pgun_pid13_constR_transformed/many_muon_hits.root";
//static std::string effq_path = "../pgun_pid13_constR_transformed/many_muon_effq.root";
// static std::string mc_path = "../web/pgun_mu_5GeV_20250708/many_muon_hits_selected.root";
// static std::string effq_path = "../web/pgun_mu_5GeV_20250708/many_muon_effq.root";
// static std::string mc_path = "../web/pgun_mu_3GeV_20250709/many_muon_hits_selected.root";
// static std::string effq_path = "../web/pgun_mu_3GeV_20250709/many_muon_effq_selected.root";
// static std::string mc_path = "../pgun_mu_20250718/pgun_20250718/filtered_pgun_3GeV_2mm_hits.root";
// static std::string effq_path = "../pgun_mu_20250718/pgun_20250718/filtered_pgun_3GeV_2mm_effq.root";
// static std::string mc_path = "../pgun_mu_20250718/pgun_20250718/filtered_pgun_3GeV_2mm_20250718_hits.root";
// static std::string effq_path = "../pgun_mu_20250718/pgun_20250718/filtered_pgun_3GeV_2mm_20250718_effq.root";

// edit 20250722; default
// static std::string mc_path = "../pgun_3GeV_20250722/pgun_mu_3GeV_2mm_20250722_hits.root";
// static std::string effq_path = "../pgun_3GeV_20250722/pgun_mu_3GeV_2mm_20250722_effq.root";
// edit 20250722; default setup bug fix; file renamed
// static std::string mc_path = "../pgun_mu_20250722/pgun_mu_3GeV_2mm_20250722_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250722/pgun_mu_3GeV_2mm_20250722_filtered_effq.root";

// edit 20250724; delay 18; bug fix
// static std::string mc_path = "../pgun_mu_20250724/pgun_mu_3GeV_2mm_20250724_delay18_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250724/pgun_mu_3GeV_2mm_20250724_delay18_filtered_effq.root";

// edit 20250728; no reset
// static std::string mc_path = "../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_noreset_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_noreset_filtered_effq.root";

// edit 20250728; delay 18, no reset
// static std::string mc_path = "../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_delay18_noreset_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250728/pgun_mu_3GeV_2mm_20250728_delay18_noreset_filtered_effq.root";

// edit 20250730; xoffset 0.5cm
// static std::string mc_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_0p5cm_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_0p5cm_filtered_effq.root";

// edit 20250730; xoffset 1.0cm
// static std::string mc_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_1p0cm_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_1p0cm_filtered_effq.root";

// edit 20250730; xoffset 2.0cm
// static std::string mc_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_2p0cm_filtered_hits.root";
// static std::string effq_path = "../pgun_mu_20250730/pgun_mu_3GeV_2mm_20250730_xoffset_2p0cm_filtered_effq.root";

// edit 20250805
static std::string data_path = "../pgun_mu_20250805/track_data.root";
static std::string mc_path = "../pgun_mu_20250805/pgun_mu_3GeV_2mm_20250805_filtered_hits.root";
static std::string effq_path = "../pgun_mu_20250805/pgun_mu_3GeV_2mm_20250805_filtered_effq.root";



TH1F* draw_from_tree(std::string filename, std::string treename, std::string var, std::string sel, int n=45, float nmin=0, float nmax=45) {
    // Load the ROOT file
    TFile *file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return nullptr;
    }

    // Get the tree from the file
    TTree *tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
        std::cerr << "Error: Tree not found!" << std::endl;
        return nullptr;
    }

    // Draw the 2D histogram
    TH1F *h1d = new TH1F("h1d", "", n, nmin, nmax);
    tree->Draw(::Form("%s>>h1d", var.c_str()), sel.c_str());
    h1d->SetDirectory(0);
    file->Close();
    return h1d;
}

double sum_total_length(std::string filename, std::string treename){
    auto df = ROOT::RDataFrame(treename, filename);
    return df.Sum("distance").GetValue();
}

void draw_dx(){
    auto hdx_2x2 = draw_from_tree(data_path, "selected_data/hits", "dx", "", 100, -2,3);
    hdx_2x2->SetName("hdx_2x2");
    auto hdx_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "dx", "", 100,-2,3);
    hdx_tred->SetName("hdx_tred");
    TCanvas* c1 = new TCanvas("cdx", "dx", 800, 600);
    hdx_2x2->SetTitle(";dx;normalized counts");
    hdx_2x2->Scale(1./hdx_2x2->GetEntries());
    hdx_tred->Scale(1./hdx_tred->GetEntries());
    hdx_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hdx_2x2->GetMaximum(), hdx_tred->GetMaximum()));
    hdx_2x2->Draw();
    hdx_2x2->SetLineColor(kRed);
    hdx_tred->Draw("SAME");
    hdx_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hdx_2x2, "2x2");
    leg->AddEntry(hdx_tred, "tred");
    leg->Draw();
    c1->Print("comp_dx.png");
}

void draw_totQ(const bool uselength, const bool useqeff=true){

    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0");
    htotQ_2x2->SetName("htotQ_2x2");
    auto htotQ_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totQ", "tindex == 0");
    htotQ_tred->SetName("htotQ_tred");
    auto htotQ_effq = draw_from_tree(effq_path, "mu_ndlar/effq", "totQ.", "tindex == 0 && totQ > 5");
    auto htotQ_effq2 = draw_from_tree(effq_path, "mu_ndlar/effq", "totQ.", "tindex == 0 && totQ > 3");
    auto htotQ_effq3 = draw_from_tree(effq_path, "mu_ndlar/effq", "totQ.", "tindex == 0 && totQ > 7");
    htotQ_effq->SetName("htotQ_effq");
    htotQ_effq2->SetName("htotQ_effq2");
    htotQ_effq3->SetName("htotQ_effq3");
    TCanvas* c1 = new TCanvas("ctotQ", "totQ", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        auto d_effq = sum_total_length(effq_path, "mu_ndlar/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
        if (useqeff && uselength) {
            htotQ_effq->Scale(1./d_effq);
            htotQ_effq2->Scale(1./d_effq);
            htotQ_effq3->Scale(1./d_effq);
        }
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);

    if (useqeff && uselength) {
        htotQ_effq->Draw("HIST SAME");
        htotQ_effq->SetLineColor(kGreen-3);
        htotQ_effq->SetLineStyle(kDashed);

        htotQ_effq2->Draw("HIST SAME");
        htotQ_effq2->SetLineColor(kGreen+3);
        htotQ_effq2->SetLineStyle(3);

        htotQ_effq3->Draw("HIST SAME");
        htotQ_effq3->SetLineColor(kGreen-5);
        htotQ_effq3->SetLineStyle(4);
    }

    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    if (useqeff && uselength) {
        leg->AddEntry(htotQ_effq, "effq; totQ>5ke-");
        leg->AddEntry(htotQ_effq2, "effq; totQ>3ke-");
        leg->AddEntry(htotQ_effq3, "effq; totQ>7ke-");
    }
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        // tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotQ_2x2->Integral(), d_2x2));
        // tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (tred): %.2f * %.0fcm", htotQ_tred->Integral(), d_tred));
    }
    if (uselength) {
        c1->Print("comp_totQ_norm_by_l.png");
    } else {
        c1->Print("comp_totQ.png");
    }
}

void draw_totN(const bool uselength){

    auto htotN_2x2 = draw_from_tree(data_path, "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotN_2x2");
    auto htotN_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_tred->SetName("htotN_tred");
    TCanvas* c2 = new TCanvas("ctotN", "totN", 800, 600);
    htotN_2x2->SetTitle("total N per pixel;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_tred->Scale(1./d_tred);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_tred->Scale(1./htotN_tred->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_tred->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_tred->Draw("SAME");
    htotN_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_tred, "tred");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (tred): %.2f * %.0fcm", htotN_tred->Integral(), d_tred));
    }
    if (uselength) {
        c2->Print("comp_totN_norm_by_l.png");
    } else {
        c2->Print("comp_totN.png");
    }
}

void draw_totN_totQ30(const bool uselength){

    auto htotN_2x2 = draw_from_tree(data_path, "selected_data/hits", "totN", "tindex == 0 && totQ>30", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotNtotQ30_2x2");
    auto htotN_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totN", "tindex == 0 && totQ>30", 5, -0.5, 4.5);
    htotN_tred->SetName("htotNtotQ30_tred");
    TCanvas* c2 = new TCanvas("ctotN_totQ30", "totN_totQ30", 800, 600);
    htotN_2x2->SetTitle("total N per pixel,totQ>30;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_tred->Scale(1./d_tred);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_tred->Scale(1./htotN_tred->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_tred->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_tred->Draw("SAME");
    htotN_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_tred, "tred");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (tred): %.2f * %.0fcm", htotN_tred->Integral(), d_tred));
    }
    if (uselength) {
        c2->Print("comp_totNtotQ30_norm_by_l.png");
    } else {
        c2->Print("comp_totNtotQ30.png");
    }
}

void draw_totQ_totN1(const bool uselength){

    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 1");
    htotQ_2x2->SetName("htotQ_2x2_totN1");
    auto htotQ_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totQ", "tindex == 0 && totN==1");
    htotQ_tred->SetName("htotQ_tred_totN1");
    TCanvas* c1 = new TCanvas("ctotQtotN1", "totQtotN1", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==1;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN1_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN1.png");
    }
}

void draw_totQ_totN2(const bool uselength){

    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 2");
    htotQ_2x2->SetName("htotQ_2x2_totN2");
    auto htotQ_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totQ", "tindex == 0 && totN==2");
    htotQ_tred->SetName("htotQ_tred_totN2");
    TCanvas* c1 = new TCanvas("ctotQtotN2", "totQtotN2", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==2;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN2_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN2.png");
    }
}

void draw_totQ_totN3(const bool uselength){

    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 3");
    htotQ_2x2->SetName("htotQ_2x2_totN3");
    auto htotQ_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "totQ", "tindex == 0 && totN==3");
    htotQ_tred->SetName("htotQ_tred_totN3");
    TCanvas* c1 = new TCanvas("ctotQtotN3", "totQtotN3", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==3;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN3_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN3.png");
    }
}

void draw_thres(const bool uselength){

    auto hthres_2x2 = draw_from_tree(data_path, "selected_data/hits", "thres", "tindex == 0");
    hthres_2x2->SetName("hthres_2x2");
    auto hthres_tred = draw_from_tree(mc_path, "mu_ndlar/hits", "thres", "tindex == 0");
    hthres_tred->SetName("hthres_tred");
    TCanvas* c1 = new TCanvas("cthres", "thres", 800, 600);
    hthres_2x2->SetTitle("frequency of thresholds at each triggered channel;thres;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "mu_ndlar/distances");
        hthres_2x2->Scale(1./d_2x2);
        hthres_tred->Scale(1./d_tred);
    } else {
        hthres_2x2->Scale(1./hthres_2x2->GetEntries());
        hthres_tred->Scale(1./hthres_tred->GetEntries());
    }
    hthres_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hthres_2x2->GetMaximum(), hthres_tred->GetMaximum()));
    hthres_2x2->Draw();
    hthres_2x2->SetLineColor(kRed);
    hthres_tred->Draw("SAME");
    hthres_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hthres_2x2, "2x2");
    leg->AddEntry(hthres_tred, "tred");
    leg->Draw();
    if (uselength) {
        c1->Print("comp_thres_norm_by_l.png");
    } else {
        c1->Print("comp_thres.png");
    }
}

void draw_totQ_totN(bool uselength=false){
    TH1::SetDefaultSumw2();
    gStyle->SetOptStat(0);
    draw_totQ(uselength);
    draw_totN(uselength);
    draw_totN_totQ30(uselength);
    draw_totQ_totN1(uselength);
    draw_totQ_totN2(uselength);
    draw_totQ_totN3(uselength);
    draw_thres(uselength);
    draw_dx();
}
