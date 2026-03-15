import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:video_player/video_player.dart';
import 'services/supabase_service.dart';
import 'services/auth_service.dart';
import 'widgets/drawer_widgets.dart';
import 'package:local_auth/local_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'utils/logging_context.dart';
import 'utils/logger.dart';
import 'utils/secure_storage.dart';
import 'supabase_config.dart';
import 'services/checkin_service.dart';
import 'screens/onboarding/display_name_screen.dart';

final supabaseService = SupabaseService();
final authService = AuthService();
final AppLogger logger = AppLogger(supabaseService);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Supabase.initialize(
    url: supabaseUrl,
    anonKey: supabaseKey,
  );
  await LoggingContext.initialize();
  CheckinService.startListening();

  runApp(const RollItBackApp());
}

class RollItBackApp extends StatelessWidget {
  const RollItBackApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Roll It Back',
      theme: ThemeData.dark(),
      home: const AuthGate(),
    );
  }
}

class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  late final Stream<AuthState> _authStream;

  @override
  void initState() {
    super.initState();
    _authStream = Supabase.instance.client.auth.onAuthStateChange;
    Future.microtask(() async {
      await _attemptBiometricLogin();
    });

    Future.microtask(() async => await _postInitAsync());
  }

  Future<void> _postInitAsync() async {
    _authStream.listen((event) async {
      final session = event.session;
      if (session == null) {
        await logger.logEvent('auth', 'User logged out');
      } else {
        await logger.logEvent('auth', 'User logged in', context: {
          'email': session.user.email,
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<AuthState>(
      stream: _authStream,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        } else if (snapshot.hasData && snapshot.data?.session != null) {
          return FutureBuilder<Map<String, dynamic>?>(
            key: ValueKey(snapshot.data!.session!.user.id),
            future: supabaseService.fetchCurrentProfile(),
            builder: (context, profileSnapshot) {
              if (profileSnapshot.connectionState == ConnectionState.waiting) {
                return const Scaffold(body: Center(child: CircularProgressIndicator()));
              }
              final profile = profileSnapshot.data;
              // null profile (trigger race) or missing display_name/home_gym_id → onboarding
              if (profile == null ||
                  profile['display_name'] == null ||
                  profile['home_gym_id'] == null) {
                return const DisplayNameScreen();
              }
              return const ClipListScreen();
            },
          );
        } else {
          return const AuthScreen();
        }
      },
    );
  }

  Future<bool> authenticateWithBiometrics() async {
    final localAuth = LocalAuthentication();
    final canCheck = await localAuth.canCheckBiometrics;
    final isAvailable = await localAuth.isDeviceSupported();

    if (!canCheck || !isAvailable) return false;

    try {
      final didAuthenticate = await localAuth.authenticate(
        localizedReason: 'Please authenticate to continue',
        options: const AuthenticationOptions(
          biometricOnly: true,
          stickyAuth: true,
        ),
      );
      return didAuthenticate;
    } catch (e) {
      await logger.logEvent('auth', 'Biometric auth error', context: {
        'error': e.toString()
      });
      return false;
    }
  }

  Future<void> _attemptBiometricLogin() async {
    final prefs = await SharedPreferences.getInstance();
    final useFingerprint = prefs.getBool('useFingerprint') ?? false;
    final staySignedIn = prefs.getBool('staySignedIn') ?? false;
    final existingUser = Supabase.instance.client.auth.currentUser;

    if (staySignedIn && existingUser != null) {
      await logger.logEvent('auth', 'Auto login via staySignedIn');
      return; // Let StreamBuilder handle routing
    }

    if (!useFingerprint) return;

    final stored = await SecureStorage.readCredentials();
    final storedEmail = stored['email'];
    final storedPassword = stored['password'];

    if (storedEmail == null || storedPassword == null) {
      await logger.logEvent('auth', 'No stored credentials for biometric login');
      return;
    }

    const maxAttempts = 5;

    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        final success = await authenticateWithBiometrics();

        if (success) {
          try {
            await Supabase.instance.client.auth.signInWithPassword(
              email: storedEmail,
              password: storedPassword,
            );

            await logger.logEvent('auth', 'Biometric login success');
            return; // Let StreamBuilder navigate to ClipListScreen
          } catch (e) {
            await logger.logEvent('auth', 'Supabase login failed after biometrics', context: {
              'error': e.toString(),
            });
            break; // Don't retry on auth error
          }
        } else {
          final remaining = maxAttempts - attempt;

          await logger.logEvent('auth', 'Biometric login failed', context: {
            'attempt': attempt,
            'remaining': remaining,
          });

          if (mounted && remaining > 0) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Fingerprint failed. You have $remaining attempt(s) left.')),
            );
          }

          // If user cancelled scan on first attempt -> force logout
          if (attempt == 1) {
            await AppDrawer.globalSignOut(context);
            return;
          }
        }
      } catch (e) {
        await logger.logEvent('auth', 'Biometric scan error', context: {
          'attempt': attempt,
          'error': e.toString(),
        });
        await AppDrawer.globalSignOut(context);
        return;
      }
    }

    // All 5 attempts failed
    await logger.logEvent('auth', 'Biometric login failed 5 times — logging out');
    await AppDrawer.globalSignOut(context);
  }
}

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});
  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  bool isSignup = false;
  bool _passwordVisible = false;

  Future<void> authenticate() async {
    final email = emailController.text.trim();
    final password = passwordController.text.trim();

    // Save credentials to SecureStorage if fingerprint is enabled
    final prefs = await SharedPreferences.getInstance();
    final useFingerprint = prefs.getBool('useFingerprint') ?? false;
    if (useFingerprint) {
      await SecureStorage.saveCredentials(email, password);
    }

    if (isSignup) {
      try {
        await authService.signUp(email: email, password: password);
      } catch (e, stack) {
        await logger.logEvent('signup', 'Signup error', context: {
          'error': e.toString(),
          'stack': stack.toString(),
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Signup error: $e")),
          );
        }
      }
    } else {
      try {
        await authService.signIn(email: email, password: password);
      } catch (e, stack) {
        await logger.logEvent('login', 'Login error', context: {
          'error': e.toString(),
          'stack': stack.toString(),
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Login error: $e")),
          );
        }
      }
    }

    final user = Supabase.instance.client.auth.currentUser;
    if (user != null) {
      await logger.logEvent('auth', 'User metadata after login', context: {
        'uid': user.id,
        'email': user.email,
      });
    }
    // AuthGate StreamBuilder handles navigation → onboarding or clips
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Roll It Back Auth')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            TextField(
              controller: emailController,
              decoration: const InputDecoration(labelText: 'Email'),
            ),
            TextField(
              controller: passwordController,
              obscureText: !_passwordVisible,
              decoration: InputDecoration(
                labelText: 'Password',
                suffixIcon: IconButton(
                  icon: Icon(_passwordVisible ? Icons.visibility : Icons.visibility_off),
                  onPressed: () {
                    setState(() {
                      _passwordVisible = !_passwordVisible;
                    });
                  },
                ),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: authenticate,
              child: Text(isSignup ? 'Sign Up' : 'Log In'),
            ),
            TextButton(
              onPressed: () => setState(() => isSignup = !isSignup),
              child: Text(isSignup ? 'Already have an account? Log in' : 'No account? Sign up'),
            ),
          ],
        ),
      ),
    );
  }
}

class ClipListScreen extends StatefulWidget {
  const ClipListScreen({super.key});
  @override
  State<ClipListScreen> createState() => _ClipListScreenState();
}

class _ClipListScreenState extends State<ClipListScreen> {
  List<ClipMetadata> clips = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    fetchUserClips();
  }

  Future<void> fetchUserClips() async {
    try {
      final profile = await supabaseService.fetchCurrentProfile();
      if (profile == null) {
        setState(() => isLoading = false);
        return;
      }
      final fetchedClips = await supabaseService.fetchMyClips(profile['id']);
      setState(() {
        clips = fetchedClips;
        isLoading = false;
      });
    } catch (e, stack) {
      await logger.logEvent('clip_list', 'Error fetching user clips', context: {
        'error': e.toString(),
        'stack': stack.toString(),
      });
      setState(() => isLoading = false);
    }
  }

  Future<void> playClip(ClipMetadata clip) async {
    try {
      final url = await supabaseService.getSignedUrl(clip.storageObjectPath);
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => ClipPlayerScreen(url: url)),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load clip: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Your Clips'),
      ),
      drawer: const AppDrawer(),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: ListView.builder(
                    itemCount: clips.length,
                    itemBuilder: (context, index) {
                      final clip = clips[index];
                      return ListTile(
                        title: Text(clip.storageObjectPath),
                        subtitle: Text('Duration: ${clip.durationSeconds}s'),
                        trailing: const Icon(Icons.play_arrow),
                        onTap: () => playClip(clip),
                      );
                    },
                  ),
                ),
              ],
            ),
    );
  }
}

class ClipPlayerScreen extends StatefulWidget {
  final String url;
  const ClipPlayerScreen({super.key, required this.url});
  @override
  State<ClipPlayerScreen> createState() => _ClipPlayerScreenState();
}

class _ClipPlayerScreenState extends State<ClipPlayerScreen> {
  late VideoPlayerController controller;

  @override
  void initState() {
    super.initState();
    controller = VideoPlayerController.networkUrl(Uri.parse(widget.url))
      ..initialize().then((_) => setState(() {}))
      ..play();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Playing Clip')),
      body: Center(
        child: controller.value.isInitialized
            ? AspectRatio(aspectRatio: controller.value.aspectRatio, child: VideoPlayer(controller))
            : const CircularProgressIndicator(),
      ),
    );
  }
}
