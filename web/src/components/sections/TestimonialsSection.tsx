'use client';

import { motion } from 'framer-motion';
import { Star, Quote } from 'lucide-react';

const testimonials = [
  {
    name: 'Rajesh Patil',
    role: 'Soybean Farmer, Pune',
    avatar: '👨‍🌾',
    content: 'MANDIMITRA helped me understand when to sell my soybean crop. The price forecast was accurate, and I earned ₹15,000 more than last year!',
    rating: 5,
  },
  {
    name: 'Sunita Deshmukh',
    role: 'Cotton Farmer, Nashik',
    avatar: '👩‍🌾',
    content: 'The crop risk alerts saved my cotton crop from pest damage. I got the warning 3 days before the problem would have spread.',
    rating: 5,
  },
  {
    name: 'Manoj Jadhav',
    role: 'Onion Farmer, Ahmednagar',
    avatar: '👨‍🌾',
    content: 'Simple to use even for someone like me who doesn\'t know much about technology. The Hindi language support is very helpful.',
    rating: 5,
  },
  {
    name: 'Priya Kulkarni',
    role: 'Wheat Farmer, Solapur',
    avatar: '👩‍🌾',
    content: 'Market comparison feature helped me find a mandi where I got ₹200 more per quintal. Transport cost calculation was very accurate.',
    rating: 5,
  },
];

export function TestimonialsSection() {
  return (
    <section className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-medium mb-4">
            Testimonials
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-900 mb-4">
            Loved by Farmers
            <br />
            <span className="text-gradient">Across Maharashtra</span>
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            See what farmers are saying about MANDIMITRA
          </p>
        </motion.div>

        {/* Testimonials Grid */}
        <div className="grid md:grid-cols-2 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="bg-gradient-to-br from-slate-50 to-emerald-50/30 rounded-2xl p-8 border border-slate-100 hover:shadow-lg transition-shadow"
            >
              {/* Quote Icon */}
              <Quote className="w-10 h-10 text-emerald-200 mb-4" />

              {/* Content */}
              <p className="text-lg text-slate-700 mb-6 leading-relaxed">
                "{testimonial.content}"
              </p>

              {/* Rating */}
              <div className="flex mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-5 h-5 text-amber-400 fill-current" />
                ))}
              </div>

              {/* Author */}
              <div className="flex items-center">
                <div className="text-4xl mr-4">{testimonial.avatar}</div>
                <div>
                  <div className="font-semibold text-slate-900">{testimonial.name}</div>
                  <div className="text-sm text-slate-500">{testimonial.role}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
