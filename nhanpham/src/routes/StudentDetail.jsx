import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

export const StudentDetail = () => {
    const { id } = useParams();

    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white">
            <div className="container mx-auto px-6 py-8">
                {/* Back Button */}
                <Link
                    to="/student"
                    className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
                >
                    <ArrowLeft className="w-5 h-5" />
                    <span>Back to Student Portal</span>
                </Link>

                {/* Content */}
                <div className="flex items-center justify-center min-h-[60vh]">
                    <div className="text-center">
                        <div className="w-20 h-20 bg-teal-500/20 rounded-full flex items-center justify-center mx-auto mb-6 border border-teal-500/30">
                            <span className="text-4xl font-bold text-teal-500">{id}</span>
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold mb-4">
                            Hello <span className="text-teal-500">{id}</span>
                        </h1>
                        <p className="text-gray-400 text-lg">
                            Student Detail Page - ID: {id}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentDetail;
