'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown,
  BarChart3, 
  Calendar, 
  MapPin, 
  AlertTriangle, 
  CheckCircle2, 
  Info,
  Loader2,
  IndianRupee,
  ArrowRight,
  Target,
  Clock,
  Sparkles
} from 'lucide-react';
import Link from 'next/link';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Legend
} from 'recharts';

interface PriceResult {
  commodity: string;
  market: string;
  current_price: number;
  price_date: string;
  price_source: string;
  forecasts: Array<{
    date: string;
    predicted_price: number;
    lower_bound: number;
    upper_bound: number;
    confidence: number;
  }>;
  price_trend: string;
  recommendation: string;
  forecast_7d: number;
  forecast_14d: number;
  forecast_30d: number;
  expected_change_percent: number;
  model_confidence: number;
}

export default function PriceForecastPage() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [selectedCommodity, setSelectedCommodity] = useState('');
  const [selectedMarket, setSelectedMarket] = useState('');
  const [forecastDays, setForecastDays] = useState(14);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PriceResult | null>(null);
  const [error, setError] = useState('');
  const [commodities, setCommodities] = useState<string[]>([]);
  const [markets, setMarkets] = useState<string[]>([]);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Fetch real commodity list from API
  useEffect(() => {
    fetch('/api/price/commodities')
      .then((r) => r.json())
      .then((data) => setCommodities(data.commodities ?? []))
      .catch(() => setCommodities(['Onion', 'Tomato', 'Soyabean', 'Wheat']));
  }, []);

  // Fetch markets filtered by selected commodity
  useEffect(() => {
    const url = selectedCommodity
      ? `/api/price/markets?commodity=${encodeURIComponent(selectedCommodity)}`
      : '/api/price/markets';
    fetch(url)
      .then((r) => r.json())
      .then((data) => {
        setMarkets(data.markets ?? []);
        // Reset market selection when commodity changes
        setSelectedMarket('');
      })
      .catch(() => setMarkets(['Pune', 'Nagpur', 'Nashik']));
  }, [selectedCommodity]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('/api/price/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          commodity: selectedCommodity,
          market: selectedMarket,
          forecast_days: forecastDays,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Forecast failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 rounded-xl shadow-lg border border-slate-100">
          <p className="font-medium text-slate-900 mb-2">{label}</p>
          <p className="text-emerald-600 text-sm">
            Predicted: {formatPrice(payload[0]?.value)}
          </p>
          {payload[1] && (
            <p className="text-slate-400 text-xs mt-1">
              Range: {formatPrice(payload[1]?.payload?.lower_bound)} - {formatPrice(payload[1]?.payload?.upper_bound)}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-amber-50/30">
      <Navbar isScrolled={isScrolled} />
      
      <section className="pt-24 sm:pt-32 pb-12 sm:pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8 sm:mb-12"
          >
            <div className="inline-flex items-center px-4 py-2 bg-amber-100 text-amber-700 rounded-full text-sm font-medium mb-4">
              <TrendingUp className="w-4 h-4 mr-2" />
              AI Price Intelligence
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900 mb-3 sm:mb-4">
              Price <span className="text-gradient-amber">Forecast</span>
            </h1>
            <p className="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto px-2">
              Get accurate price predictions with confidence intervals to make 
              better selling decisions.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-5 gap-6 lg:gap-8">
            {/* Input Form - 2 cols */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-2"
            >
              <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-5 sm:p-8 border border-slate-100 lg:sticky lg:top-24">
                <h2 className="text-xl font-semibold text-slate-900 mb-6 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-amber-500" />
                  Forecast Settings
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Commodity Selection */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Commodity
                    </label>
                    <select
                      value={selectedCommodity}
                      onChange={(e) => setSelectedCommodity(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all"
                      required
                    >
                      <option value="">Select commodity...</option>
                      {commodities.map((commodity) => (
                        <option key={commodity} value={commodity}>{commodity}</option>
                      ))}
                    </select>
                  </div>

                  {/* Market Selection */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      <MapPin className="w-4 h-4 inline mr-1" />
                      Market
                    </label>
                    <select
                      value={selectedMarket}
                      onChange={(e) => setSelectedMarket(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all"
                      required
                    >
                      <option value="">Select market...</option>
                      {markets.map((market) => (
                        <option key={market} value={market}>{market}</option>
                      ))}
                    </select>
                  </div>

                  {/* Forecast Days */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      <Clock className="w-4 h-4 inline mr-1" />
                      Forecast Period
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {[7, 14, 15].map((days) => (
                        <button
                          key={days}
                          type="button"
                          onClick={() => setForecastDays(days)}
                          className={`px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                            forecastDays === days
                              ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/30'
                              : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          {days} Days
                        </button>
                      ))}
                    </div>
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
                    className="w-full px-6 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-xl hover:from-amber-600 hover:to-orange-600 transition-all shadow-lg shadow-amber-500/30 hover:shadow-amber-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Forecasting...
                      </>
                    ) : (
                      <>
                        Generate Forecast
                        <Sparkles className="w-5 h-5 ml-2" />
                      </>
                    )}
                  </button>
                </form>
              </div>
            </motion.div>

            {/* Results Panel - 3 cols */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="lg:col-span-3"
            >
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="space-y-6"
                  >
                    {/* Summary Cards */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                      <div className="bg-white rounded-xl p-3 sm:p-4 shadow-lg shadow-slate-200/50 border border-slate-100">
                        <div className="text-xs sm:text-sm text-slate-500 mb-1">Current Price</div>
                        <div className="text-lg sm:text-xl font-bold text-slate-900">{formatPrice(result.current_price)}</div>
                        <div className="text-xs text-slate-400 mt-0.5">/quintal</div>
                        <div className="flex items-center gap-1 mt-1">
                          <span className={`inline-block w-1.5 h-1.5 rounded-full ${result.price_source === 'live' ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
                          <span className="text-[10px] text-slate-400">
                            {result.price_date}{result.price_source === 'live' ? ' • Live' : ''}
                          </span>
                        </div>
                      </div>
                      <div className="bg-white rounded-xl p-3 sm:p-4 shadow-lg shadow-slate-200/50 border border-slate-100">
                        <div className="text-xs sm:text-sm text-slate-500 mb-1">7-Day Forecast</div>
                        <div className="text-lg sm:text-xl font-bold text-slate-900">{formatPrice(result.forecast_7d)}</div>
                        <div className={`text-xs mt-1 ${result.forecast_7d > result.current_price ? 'text-emerald-500' : 'text-red-500'}`}>
                          {result.forecast_7d > result.current_price ? '↑' : '↓'} 
                          {Math.abs(((result.forecast_7d - result.current_price) / result.current_price) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-white rounded-xl p-3 sm:p-4 shadow-lg shadow-slate-200/50 border border-slate-100">
                        <div className="text-xs sm:text-sm text-slate-500 mb-1">14-Day Forecast</div>
                        <div className="text-lg sm:text-xl font-bold text-slate-900">{formatPrice(result.forecast_14d)}</div>
                        <div className={`text-xs mt-1 ${result.forecast_14d > result.current_price ? 'text-emerald-500' : 'text-red-500'}`}>
                          {result.forecast_14d > result.current_price ? '↑' : '↓'} 
                          {Math.abs(((result.forecast_14d - result.current_price) / result.current_price) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className={`rounded-xl p-3 sm:p-4 shadow-lg shadow-slate-200/50 border ${
                        result.recommendation === 'HOLD' 
                          ? 'bg-emerald-50 border-emerald-200' 
                          : 'bg-amber-50 border-amber-200'
                      }`}>
                        <div className="text-xs sm:text-sm text-slate-500 mb-1">Recommendation</div>
                        <div className={`text-lg sm:text-xl font-bold ${
                          result.recommendation === 'HOLD' ? 'text-emerald-600' : 'text-amber-600'
                        }`}>
                          {result.recommendation}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {result.model_confidence.toFixed(0)}% confidence
                        </div>
                      </div>
                    </div>

                    {/* Chart */}
                    <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-4 sm:p-6 border border-slate-100">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 sm:mb-6">
                        <div className="min-w-0">
                          <h3 className="text-base sm:text-lg font-semibold text-slate-900 truncate">
                            {result.commodity} Price Forecast
                          </h3>
                          <p className="text-sm text-slate-500">{result.market} Market</p>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                          result.price_trend === 'up' 
                            ? 'bg-emerald-100 text-emerald-700' 
                            : result.price_trend === 'down'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-slate-100 text-slate-700'
                        }`}>
                          {result.price_trend === 'up' && <TrendingUp className="w-4 h-4 inline mr-1" />}
                          {result.price_trend === 'down' && <TrendingDown className="w-4 h-4 inline mr-1" />}
                          {result.expected_change_percent > 0 ? '+' : ''}{result.expected_change_percent.toFixed(1)}%
                        </div>
                      </div>

                      <div className="h-[220px] sm:h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={result.forecasts} margin={{ left: -10, right: 5, top: 5, bottom: 5 }}>
                            <defs>
                              <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2}/>
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis 
                              dataKey="date" 
                              tick={{ fontSize: 10, fill: '#64748b' }}
                              tickLine={false}
                              interval="preserveStartEnd"
                            />
                            <YAxis 
                              tick={{ fontSize: 10, fill: '#64748b' }}
                              tickLine={false}
                              tickFormatter={(value) => `₹${value}`}
                              width={55}
                              domain={['dataMin - 100', 'dataMax + 100']}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                              type="monotone"
                              dataKey="upper_bound"
                              stroke="transparent"
                              fill="url(#colorConfidence)"
                              fillOpacity={1}
                            />
                            <Area
                              type="monotone"
                              dataKey="lower_bound"
                              stroke="transparent"
                              fill="white"
                              fillOpacity={1}
                            />
                            <Line
                              type="monotone"
                              dataKey="predicted_price"
                              stroke="#f59e0b"
                              strokeWidth={3}
                              dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
                              activeDot={{ r: 6, strokeWidth: 0 }}
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 mt-4 text-xs sm:text-sm">
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-amber-500 rounded-full mr-2" />
                          <span className="text-slate-600">Predicted Price</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-amber-200 rounded-full mr-2" />
                          <span className="text-slate-600">Confidence Interval</span>
                        </div>
                      </div>
                    </div>

                    {/* Forecast Table */}
                    <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-4 sm:p-6 border border-slate-100">
                      <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-4">
                        Daily Forecasts
                      </h3>
                      <div className="overflow-x-auto -mx-2 sm:mx-0">
                        <table className="w-full min-w-[480px]">
                          <thead>
                            <tr className="text-left text-xs sm:text-sm text-slate-500 border-b border-slate-100">
                              <th className="pb-3 pl-2 sm:pl-0 font-medium">Date</th>
                              <th className="pb-3 font-medium">Predicted</th>
                              <th className="pb-3 font-medium">Range</th>
                              <th className="pb-3 font-medium">Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.forecasts.map((forecast, index) => (
                              <tr key={index} className="border-b border-slate-50 text-xs sm:text-sm">
                                <td className="py-2.5 sm:py-3 pl-2 sm:pl-0 text-slate-600 whitespace-nowrap">{forecast.date}</td>
                                <td className="py-2.5 sm:py-3 font-medium text-slate-900 whitespace-nowrap">
                                  {formatPrice(forecast.predicted_price)}
                                </td>
                                <td className="py-2.5 sm:py-3 text-slate-500 whitespace-nowrap">
                                  {formatPrice(forecast.lower_bound)} - {formatPrice(forecast.upper_bound)}
                                </td>
                                <td className="py-2.5 sm:py-3">
                                  <div className="flex items-center">
                                    <div className="w-12 sm:w-16 h-2 bg-slate-100 rounded-full overflow-hidden mr-2 flex-shrink-0">
                                      <div 
                                        className="h-full bg-amber-500 rounded-full"
                                        style={{ width: `${forecast.confidence}%` }}
                                      />
                                    </div>
                                    <span className="text-slate-500">{forecast.confidence.toFixed(0)}%</span>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="bg-gradient-to-br from-slate-100 to-amber-50/50 rounded-2xl p-8 sm:p-12 border border-slate-200 h-full min-h-[300px] sm:min-h-[500px] flex flex-col items-center justify-center text-center"
                  >
                    <div className="w-20 h-20 bg-amber-100 rounded-2xl flex items-center justify-center mb-6">
                      <TrendingUp className="w-10 h-10 text-amber-500" />
                    </div>
                    <h3 className="text-xl font-semibold text-slate-900 mb-2">
                      Price Forecast Results
                    </h3>
                    <p className="text-slate-500 max-w-sm mb-6">
                      Select a commodity and market, then click "Generate Forecast" to see 
                      AI-powered price predictions with confidence intervals.
                    </p>
                    <div className="flex items-center gap-4 text-sm text-slate-400">
                      <div className="flex items-center">
                        <Target className="w-4 h-4 mr-1" />
                        93% Accuracy
                      </div>
                      <div className="flex items-center">
                        <Clock className="w-4 h-4 mr-1" />
                        30-Day Forecast
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </section>

      <Footer />

      <style jsx global>{`
        .text-gradient-amber {
          background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
      `}</style>
    </main>
  );
}
