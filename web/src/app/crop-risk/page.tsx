'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Leaf, 
  Calendar, 
  MapPin, 
  AlertTriangle, 
  CheckCircle2, 
  Info,
  Loader2,
  TrendingUp,
  Droplets,
  Thermometer,
  Wind,
  ArrowRight
} from 'lucide-react';
import Link from 'next/link';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';

interface CropRiskResult {
  crop: string;
  district: string;
  current_stage: string;
  days_since_sowing: number;
  risk_level: string;
  risk_score: number;
  ml_probabilities: { low: number; medium: number; high: number };
  risk_factors: string[];
  recommendations: string[];
  weather_summary: Record<string, number>;
}

export default function CropRiskPage() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [selectedCrop, setSelectedCrop] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [sowingDate, setSowingDate] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<CropRiskResult | null>(null);
  const [error, setError] = useState('');
  const [crops, setCrops] = useState<string[]>([]);
  const [districts, setDistricts] = useState<string[]>([]);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Fetch real crop and district lists from API
  useEffect(() => {
    fetch('/api/crop-risk/crops')
      .then((r) => r.json())
      .then((data) => setCrops(data.crops ?? []))
      .catch(() => setCrops(['Soybean', 'Cotton', 'Tur', 'Urad', 'Onion', 'Wheat', 'Jowar', 'Bajra', 'Maize', 'Gram', 'Tomato']));
    fetch('/api/crop-risk/districts')
      .then((r) => r.json())
      .then((data) => setDistricts(data.districts ?? []))
      .catch(() => setDistricts(['Pune', 'Nashik', 'Nagpur', 'Solapur', 'Kolhapur']));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('/api/crop-risk/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop: selectedCrop,
          district: selectedDistrict,
          sowing_date: sowingDate,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Assessment failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'high': return 'from-red-500 to-red-600';
      case 'medium': return 'from-amber-500 to-orange-500';
      case 'low': return 'from-emerald-500 to-emerald-600';
      default: return 'from-slate-500 to-slate-600';
    }
  };

  const getRiskBg = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-700 border-red-200';
      case 'medium': return 'bg-amber-100 text-amber-700 border-amber-200';
      case 'low': return 'bg-emerald-100 text-emerald-700 border-emerald-200';
      default: return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-emerald-50/30">
      <Navbar isScrolled={isScrolled} />
      
      <section className="pt-24 sm:pt-32 pb-12 sm:pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8 sm:mb-12"
          >
            <div className="inline-flex items-center px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-medium mb-4">
              <Shield className="w-4 h-4 mr-2" />
              AI-Powered Risk Assessment
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900 mb-3 sm:mb-4">
              Crop Risk <span className="text-gradient">Advisor</span>
            </h1>
            <p className="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto px-2">
              Get instant AI-powered risk assessment for your crops based on weather, 
              growth stage, and historical patterns.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-2 gap-6 lg:gap-8">
            {/* Input Form */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-5 sm:p-8 border border-slate-100">
                <h2 className="text-xl font-semibold text-slate-900 mb-6 flex items-center">
                  <Leaf className="w-5 h-5 mr-2 text-emerald-500" />
                  Enter Crop Details
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Crop Selection */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Select Crop
                    </label>
                    <select
                      value={selectedCrop}
                      onChange={(e) => setSelectedCrop(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                      required
                    >
                      <option value="">Choose a crop...</option>
                      {crops.map((crop) => (
                        <option key={crop} value={crop}>{crop}</option>
                      ))}
                    </select>
                  </div>

                  {/* District Selection */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      <MapPin className="w-4 h-4 inline mr-1" />
                      District
                    </label>
                    <select
                      value={selectedDistrict}
                      onChange={(e) => setSelectedDistrict(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                      required
                    >
                      <option value="">Choose a district...</option>
                      {districts.map((district) => (
                        <option key={district} value={district}>{district}</option>
                      ))}
                    </select>
                  </div>

                  {/* Sowing Date */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      <Calendar className="w-4 h-4 inline mr-1" />
                      Sowing Date
                    </label>
                    <input
                      type="date"
                      value={sowingDate}
                      onChange={(e) => setSowingDate(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                      required
                    />
                  </div>

                  {/* Error Message */}
                  {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
                      <AlertTriangle className="w-4 h-4 inline mr-2" />
                      {error}
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full px-6 py-4 bg-gradient-to-r from-emerald-600 to-emerald-500 text-white font-semibold rounded-xl hover:from-emerald-700 hover:to-emerald-600 transition-all shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        Assess Crop Risk
                        <ArrowRight className="w-5 h-5 ml-2" />
                      </>
                    )}
                  </button>
                </form>

                {/* Info Box */}
                <div className="mt-6 p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="flex items-start">
                    <Info className="w-5 h-5 text-slate-400 mr-3 mt-0.5" />
                    <div className="text-sm text-slate-600">
                      <p className="font-medium text-slate-700 mb-1">How it works</p>
                      <p>Our AI analyzes weather forecasts, crop growth stages, and historical data to predict potential risks and provide actionable recommendations.</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Results Panel */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-5 sm:p-8 border border-slate-100"
                  >
                    {/* Risk Level Header */}
                    <div className="flex items-start sm:items-center justify-between gap-3 mb-6">
                      <div className="min-w-0">
                        <h3 className="text-lg sm:text-xl font-semibold text-slate-900 truncate">{result.crop}</h3>
                        <p className="text-sm sm:text-base text-slate-500 truncate">{result.district} • {result.current_stage}</p>
                      </div>
                      <div className={`px-3 sm:px-4 py-1.5 sm:py-2 rounded-full text-xs sm:text-sm font-semibold border whitespace-nowrap flex-shrink-0 ${getRiskBg(result.risk_level)}`}>
                        {result.risk_level} Risk
                      </div>
                    </div>

                    {/* Risk Meter */}
                    <div className="mb-6">
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-500">Risk Score</span>
                        <span className="font-semibold text-slate-900">{Math.round(result.risk_score)}/100</span>
                      </div>
                      <div className="h-4 bg-slate-100 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${result.risk_score}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className={`h-full bg-gradient-to-r ${getRiskColor(result.risk_level)} rounded-full`}
                        />
                      </div>
                    </div>

                    {/* Probability Cards */}
                    <div className="grid grid-cols-3 gap-2 sm:gap-3 mb-6">
                      <div className="text-center p-3 sm:p-4 bg-emerald-50 rounded-xl">
                        <div className="text-xl sm:text-2xl font-bold text-emerald-600">
                          {(result.ml_probabilities.low * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-500">Low Risk</div>
                      </div>
                      <div className="text-center p-3 sm:p-4 bg-amber-50 rounded-xl">
                        <div className="text-xl sm:text-2xl font-bold text-amber-600">
                          {(result.ml_probabilities.medium * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-500">Medium</div>
                      </div>
                      <div className="text-center p-3 sm:p-4 bg-red-50 rounded-xl">
                        <div className="text-xl sm:text-2xl font-bold text-red-600">
                          {(result.ml_probabilities.high * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-500">High Risk</div>
                      </div>
                    </div>

                    {/* Days Since Sowing */}
                    <div className="p-4 bg-slate-50 rounded-xl mb-6">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-600">Days Since Sowing</span>
                        <span className="font-semibold text-slate-900">{result.days_since_sowing} days</span>
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div>
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center">
                        <CheckCircle2 className="w-5 h-5 mr-2 text-emerald-500" />
                        Recommendations
                      </h4>
                      <div className="space-y-2">
                        {result.recommendations.slice(0, 4).map((rec, index) => (
                          <div
                            key={index}
                            className="p-3 bg-slate-50 rounded-xl text-sm text-slate-700"
                          >
                            {rec}
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="bg-gradient-to-br from-slate-100 to-emerald-50/50 rounded-2xl p-8 border border-slate-200 h-full flex flex-col items-center justify-center text-center"
                  >
                    <div className="w-20 h-20 bg-emerald-100 rounded-2xl flex items-center justify-center mb-6">
                      <Shield className="w-10 h-10 text-emerald-500" />
                    </div>
                    <h3 className="text-xl font-semibold text-slate-900 mb-2">
                      Your Risk Assessment
                    </h3>
                    <p className="text-slate-500 max-w-sm">
                      Fill in your crop details and click "Assess Crop Risk" to get 
                      AI-powered insights and recommendations.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
