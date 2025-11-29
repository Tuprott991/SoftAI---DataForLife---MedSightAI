import { useNavigate } from 'react-router-dom';
import { useAuth } from '../components/authentication';
import { LoginForm } from '../components/authentication';
import { useEffect } from 'react';

const Login = () => {
    const { login, isAuthenticated } = useAuth();
    const navigate = useNavigate();

    // Redirect if already authenticated
    useEffect(() => {
        if (isAuthenticated) {
            navigate('/home');
        }
    }, [isAuthenticated, navigate]);

    const handleLogin = (userData) => {
        login(userData);
        navigate('/home');
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center px-4">
            <div className="w-full max-w-md">
                {/* Logo/Brand Section */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-20 h-20 bg-linear-to-br from-teal-500 to-teal-600 rounded-2xl mb-4 shadow-lg shadow-teal-500/20">
                        <svg
                            className="w-12 h-12 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                            />
                        </svg>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-2">MedSightAI</h1>
                    <p className="text-gray-400">Hệ thống hỗ trợ chẩn đoán y tế thông minh</p>
                </div>

                {/* Login Card */}
                <div className="bg-[#141414] rounded-2xl shadow-2xl border border-white/5 p-8">
                    <h2 className="text-2xl font-bold text-white mb-6">Đăng nhập</h2>
                    <LoginForm onLogin={handleLogin} />
                </div>

                {/* Footer */}
                <div className="text-center mt-8">
                    <p className="text-sm text-gray-500">
                        © 2024 MedSightAI. Bảo mật và riêng tư được đảm bảo.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Login;
